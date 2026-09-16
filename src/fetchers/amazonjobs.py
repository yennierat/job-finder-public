"""amazon.jobs adapter.

Amazon runs its own system rather than a third-party ATS, so this covers exactly
one employer — but it is one of Singapore's largest tech employers and has a
real internship pipeline.

The board is a comma-separated list of SEARCH QUERIES, not a company slug:
board: "intern,internship,student". Amazon publishes ~10,000 postings globally
and its location parameters are ignored (`country[]=SGP` cheerfully returns
Bengaluru and Newcastle), so narrowing by keyword is the only way to keep the
fetch sane. The prefilter then handles location as usual.

Multiple queries matter more than they look: "intern" returns 190 results while
"internship" returns 2528, and neither is a superset of the other. Results are
deduplicated on job id across queries.

Quirks:
  - result_limit caps at 100; 200 returns an empty list rather than an error.
  - `is_intern` is present in the schema but null in search results, so
    employment type still comes from the title.
"""

import time

import requests

from src.models import Posting
from src.normalise import description_excerpt, employment_type, finalise, html_to_text

SEARCH = "https://www.amazon.jobs/en/search.json"
JOB_BASE = "https://www.amazon.jobs"
PAGE = 100        # hard cap; 200 silently returns nothing
MAX_PAGES = 40


def fetch(source_id: str, board: str) -> list[Posting]:
    queries = [q.strip() for q in board.split(",") if q.strip()] or ["internship"]
    postings: list[Posting] = []
    seen: set[str] = set()
    for query in queries:
        _fetch_query(source_id, query, postings, seen)
    return postings


def _fetch_query(source_id: str, query: str, postings: list, seen: set) -> None:
    from src.fetchers import HEADERS, POLITE_DELAY, TIMEOUT

    headers = {**HEADERS, "Accept": "application/json"}
    offset = 0

    for _ in range(MAX_PAGES):
        r = requests.get(SEARCH, headers=headers, timeout=TIMEOUT,
                         params={"base_query": query, "result_limit": PAGE,
                                 "offset": offset, "sort": "recent"})
        r.raise_for_status()
        body = r.json()
        page = body.get("jobs") or []
        if not page:
            break

        for job in page:
            job_id = str(job.get("id_icims") or job.get("id") or "")
            if not job_id or job_id in seen:
                continue
            seen.add(job_id)

            title = job.get("title") or "(untitled)"
            path = job.get("job_path") or ""
            body_text = " ".join(filter(None, [
                job.get("description_short"),
                job.get("basic_qualifications"),
                job.get("preferred_qualifications"),
                job.get("description"),
            ]))

            postings.append(finalise(Posting(
                source_id=source_id,
                external_id=job_id,
                title=title,
                url=f"{JOB_BASE}{path}" if path else None,
                # normalized_location is "Singapore, Singapore, SGP"; the plain
                # `location` field is a terser code string.
                location_raw=job.get("normalized_location") or job.get("location"),
                employment_type=employment_type(title),
                description=description_excerpt(html_to_text(body_text)),
            )))

        # Queries overlap heavily, so an all-duplicate page is normal here and
        # is NOT an end-of-results signal — only an empty page or the hit count
        # ends the loop.
        offset += PAGE
        if offset >= (body.get("hits") or 0):
            break
        time.sleep(POLITE_DELAY)
