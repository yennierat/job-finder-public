"""Ashby job-board API adapter."""

import time

import requests

from src.models import Posting
from src.normalise import description_excerpt, employment_type, finalise, html_to_text

URL = "https://api.ashbyhq.com/posting-api/job-board/{board}"

# Ashby's own employmentType vocabulary -> ours.
TYPES = {
    "fulltime": "full_time",
    "parttime": "contract",
    "intern": "internship",
    "internship": "internship",
    "contractor": "contract",
    "contract": "contract",
    "temporary": "contract",
    "graduate": "graduate_programme",
}


def fetch(source_id: str, board: str) -> list[Posting]:
    from src.fetchers import HEADERS, POLITE_DELAY, TIMEOUT

    r = requests.get(URL.format(board=board), headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    time.sleep(POLITE_DELAY)

    postings = []
    for job in r.json().get("jobs", []):
        title = job.get("title") or "(untitled)"
        primary = job.get("location")
        secondary = [s.get("location") for s in (job.get("secondaryLocations") or [])
                     if s.get("location")]
        location_raw = ("; ".join([primary, *secondary]) if primary
                        else "; ".join(secondary))

        mapped = TYPES.get((job.get("employmentType") or "").lower())
        postings.append(finalise(Posting(
            source_id=source_id,
            external_id=str(job.get("id")),
            title=title,
            url=job.get("jobUrl"),
            location_raw=location_raw or None,
            employment_type=mapped or employment_type(title),
            remote=bool(job.get("isRemote")),
            # Ashby ships the full description in the list response already.
            description=description_excerpt(html_to_text(job.get("descriptionHtml"))),
        )))
    return postings
