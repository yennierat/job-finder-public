"""Oracle Cloud (HCM Recruiting / Candidate Experience) adapter.

Where the big banks that are not on Workday live — JPMorgan runs ~7,000
postings here, including the Singapore and London 2027 summer analyst
programmes that no other adapter can see.

A source names host and site: board: "jpmc.fa.oraclecloud.com/CX_1001".
Find both by opening the employer's careers page and watching for a request to
`/hcmRestApi/resources/latest/recruitingCEJobRequisitions`.

Notes on the API:
  - `limit` silently caps at 200; asking for 500 returns 200, so paging must be
    driven by the returned count rather than by trusting the request.
  - The list response exposes no employment-type field at all (JobType,
    ContractType and WorkerType are all null), so type comes from the title —
    which is why "Summer Analyst" had to be taught to normalise.py.
  - ShortDescriptionStr is a ~100 character teaser, not the full advert.
"""

import time

import requests

from src.models import Posting
from src.normalise import description_excerpt, employment_type, finalise, html_to_text

API = ("https://{host}/hcmRestApi/resources/latest/recruitingCEJobRequisitions"
       "?onlyData=true&expand=requisitionList"
       "&finder=findReqs;siteNumber={site},limit={limit},offset={offset}")
JOB = "https://{host}/hcmUI/CandidateExperience/en/sites/{site}/job/{job_id}"
PAGE = 200          # hard cap; larger values are silently clamped
MAX_PAGES = 60      # 12000 postings, and a guard against a pagination loop


def _date_or_none(value) -> str | None:
    """Oracle returns the STRING "None" for an unset date, not a null."""
    text = str(value or "").strip()
    return text if text and text.lower() != "none" else None


def parse_board(board: str) -> tuple[str, str]:
    """"jpmc.fa.oraclecloud.com/CX_1001" -> (host, site)."""
    host, _, site = board.partition("/")
    if not host or not site:
        raise ValueError(
            f"oraclecloud board must be 'host/siteNumber', got {board!r}")
    return host, site


def fetch(source_id: str, board: str) -> list[Posting]:
    from src.fetchers import HEADERS, POLITE_DELAY, TIMEOUT

    host, site = parse_board(board)
    headers = {**HEADERS, "Accept": "application/json"}

    postings: list[Posting] = []
    seen: set[str] = set()
    offset = 0

    for _ in range(MAX_PAGES):
        url = API.format(host=host, site=site, limit=PAGE, offset=offset)
        r = requests.get(url, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        items = r.json().get("items") or []
        if not items:
            break

        page = items[0].get("requisitionList") or []
        if not page:
            break

        fresh = 0
        for job in page:
            job_id = str(job.get("Id") or "")
            if not job_id or job_id in seen:
                continue
            seen.add(job_id)
            fresh += 1

            title = job.get("Title") or "(untitled)"
            body = " ".join(filter(None, [
                job.get("ShortDescriptionStr"),
                job.get("ExternalResponsibilitiesStr"),
                job.get("ExternalQualificationsStr"),
            ]))

            postings.append(finalise(Posting(
                source_id=source_id,
                external_id=job_id,
                title=title,
                url=JOB.format(host=host, site=site, job_id=job_id),
                location_raw=job.get("PrimaryLocation"),
                # No structured employment type in this API — title only.
                employment_type=employment_type(title),
                remote=(job.get("WorkplaceTypeCode") or "").upper() == "REMOTE",
                description=description_excerpt(html_to_text(body)),
                # Oracle defines a closing date, but populating it is optional
                # and JPMorgan leaves it empty on all 600 of its requisitions.
                # Mapped anyway: it costs nothing, and a tenant that does fill
                # it in should not silently lose the date.
                deadline=_date_or_none(job.get("PostingEndDate")),
            )))

        if fresh == 0:
            break
        offset += PAGE
        total = items[0].get("TotalJobsCount") or 0
        if total and offset >= total:
            break
        time.sleep(POLITE_DELAY)

    return postings
