"""Workday CXS adapter.

Workday differs from the other platforms in three ways that shape this code:
  - It is a POST API, and the page size is hard-capped at 20 (50 or 100 is
    rejected outright with an errorCode, not silently clamped).
  - Tenant, data-centre number and site name all vary per customer, so a source
    names all three: board: "ms/wd5/External".
  - It exposes no employment-type field, so type comes from the title alone.

Pagination terminates on an empty page rather than on reaching `total`, because
that figure disagrees with reality on some tenants.
"""

import time

import requests

from src.models import Posting
from src.normalise import employment_type, finalise, html_to_text

API = "https://{tenant}.{wd}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"
JOB = "https://{tenant}.{wd}.myworkdayjobs.com/en-US/{site}{path}"
PAGE = 20          # hard cap; larger values are rejected
MAX_PAGES = 150    # 3000 postings, and a guard against a pagination loop


def parse_board(board: str) -> tuple[str, str, str]:
    """"ms/wd5/External" -> ("ms", "wd5", "External")."""
    parts = board.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"workday board must be 'tenant/wdN/site', got {board!r}")
    return parts[0], parts[1], parts[2]


def fetch(source_id: str, board: str) -> list[Posting]:
    from src.fetchers import HEADERS, POLITE_DELAY, TIMEOUT

    tenant, wd, site = parse_board(board)
    url = API.format(tenant=tenant, wd=wd, site=site)
    headers = {**HEADERS, "Content-Type": "application/json",
               "Accept": "application/json"}

    postings: list[Posting] = []
    seen: set[str] = set()
    total: int | None = None
    offset = 0

    for _ in range(MAX_PAGES):
        r = requests.post(url, headers=headers, timeout=TIMEOUT,
                          json={"appliedFacets": {}, "limit": PAGE,
                                "offset": offset, "searchText": ""})
        r.raise_for_status()
        body = r.json()
        if total is None:
            total = body.get("total") or 0

        page = body.get("jobPostings") or []
        if not page:
            break

        fresh = 0
        for job in page:
            title = job.get("title") or "(untitled)"
            path = job.get("externalPath") or ""
            external_id = _external_id(job, path, title)
            if external_id in seen:
                continue
            seen.add(external_id)
            fresh += 1

            postings.append(finalise(Posting(
                source_id=source_id,
                external_id=external_id,
                title=title,
                url=(JOB.format(tenant=tenant, wd=wd, site=site, path=path)
                     if path else None),
                location_raw=job.get("locationsText"),
                employment_type=employment_type(title),
            )))

        # Past the last page some tenants wrap and serve page 1 forever instead
        # of returning an empty list, so an all-duplicate page is the real end.
        if fresh == 0:
            break

        offset += PAGE
        if total and offset >= total:
            break
        time.sleep(POLITE_DELAY)

    return postings


def detail(source_id: str, board: str,
           posting: Posting) -> tuple[str | None, str | None]:
    """Fetch one posting's advert text and its structured closing date.

    The list endpoint returns no description at all, so without this every
    Workday posting — a third of everything that reaches the classifier — is
    judged on its job title alone.

    Two things make this work where the obvious call 403s: the CXS detail path
    is the externalPath appended to the same /wday/cxs/ prefix used for the
    list, and a Referer naming the human-facing page is required. Some tenants
    (UOB, Citi) refuse regardless; those return (None, None) and keep the
    title-only behaviour rather than failing the run.
    """
    from src.fetchers import HEADERS, POLITE_DELAY, TIMEOUT

    # The list path sleeps between pages; this path is called once per posting
    # from a thread pool, so without a delay one tenant takes ~190 concurrent
    # requests in a burst. That is how a working adapter earns a 429 or a block.
    time.sleep(POLITE_DELAY)

    tenant, wd, site = parse_board(board)
    host = f"https://{tenant}.{wd}.myworkdayjobs.com"
    path = posting.external_id if posting.external_id.startswith("/") else ""
    if not path:
        return None, None

    r = requests.get(f"{host}/wday/cxs/{tenant}/{site}{path}",
                     headers={**HEADERS, "Accept": "application/json",
                              "Referer": f"{host}/en-US/{site}{path}"},
                     timeout=TIMEOUT)
    if not r.ok:
        return None, None

    info = (r.json() or {}).get("jobPostingInfo") or {}
    # endDate is Workday's own "applications close" field. Preferred over
    # anything parsed from prose: it is structured, and it is exact.
    return html_to_text(info.get("jobDescription")), info.get("endDate") or None


def _external_id(job: dict, path: str, title: str) -> str:
    """A per-posting id that is actually unique.

    bulletFields looks like a requisition number on some tenants and is exactly
    that on many — but it is a tenant-configured display field, and UOB puts the
    LOCATION there, collapsing 1000 postings onto the id "Singapore". Only
    externalPath is reliably one-per-posting.
    """
    if path:
        return path
    bullets = job.get("bulletFields") or []
    if bullets:
        return str(bullets[0])
    return f"{title}|{job.get('locationsText') or ''}"
