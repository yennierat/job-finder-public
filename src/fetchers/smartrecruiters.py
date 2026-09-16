"""SmartRecruiters posting API adapter.

Two traps this API sets, both handled here:
  - It returns HTTP 200 with totalFound=0 for a company that does not exist, so
    a 200 alone never proves a board is real.
  - limit is capped at 100, so anything larger must be paged or silently
    truncates.
"""

import time

import requests

from src.models import Posting
from src.normalise import employment_type, finalise, html_to_text

URL = "https://api.smartrecruiters.com/v1/companies/{board}/postings"
DETAIL = "https://api.smartrecruiters.com/v1/companies/{board}/postings/{job_id}"
APPLY_URL = "https://jobs.smartrecruiters.com/{company}/{job_id}"
PAGE = 100
MAX_PAGES = 40  # 4000 postings; guards against a pagination bug looping forever


def fetch(source_id: str, board: str) -> list[Posting]:
    from src.fetchers import HEADERS, POLITE_DELAY, TIMEOUT

    postings: list[Posting] = []
    offset = 0

    for _ in range(MAX_PAGES):
        r = requests.get(URL.format(board=board), headers=HEADERS, timeout=TIMEOUT,
                         params={"limit": PAGE, "offset": offset})
        r.raise_for_status()
        body = r.json()
        page = body.get("content") or []
        if not page:
            break  # terminate on an empty page, never on reaching totalFound

        for job in page:
            loc = job.get("location") or {}
            city = loc.get("city")
            country = (loc.get("country") or "").upper()
            location_raw = (", ".join(p for p in (city, country) if p)
                            or loc.get("fullLocation"))

            company = (job.get("company") or {}).get("identifier") or board
            job_id = str(job.get("id"))
            title = job.get("name") or "(untitled)"

            postings.append(finalise(Posting(
                source_id=source_id,
                external_id=job_id,
                title=title,
                url=APPLY_URL.format(company=company, job_id=job_id),
                location_raw=location_raw,
                employment_type=employment_type(
                    title, (job.get("typeOfEmployment") or {}).get("label")),
                remote=bool(loc.get("remote")),
            )))

        offset += PAGE
        if offset >= body.get("totalFound", 0):
            break
        time.sleep(POLITE_DELAY)

    return postings


# Sections in the order they matter to a classifier: what the job is, then what
# it requires. companyDescription is skipped deliberately — it is boilerplate
# marketing repeated on every posting, and it would dominate the excerpt.
DETAIL_SECTIONS = ("jobDescription", "qualifications", "additionalInformation")


def detail(source_id: str, board: str,
           posting: Posting) -> tuple[str | None, str | None]:
    """Fetch one posting's advert text. SmartRecruiters exposes no closing date."""
    from src.fetchers import HEADERS, POLITE_DELAY, TIMEOUT

    time.sleep(POLITE_DELAY)  # called once per posting, from a thread pool
    r = requests.get(DETAIL.format(board=board, job_id=posting.external_id),
                     headers=HEADERS, timeout=TIMEOUT)
    if not r.ok:
        return None, None

    sections = ((r.json() or {}).get("jobAd") or {}).get("sections") or {}
    parts = [html_to_text((sections.get(name) or {}).get("text"))
             for name in DETAIL_SECTIONS]
    text = " ".join(p for p in parts if p)
    return text or None, None
