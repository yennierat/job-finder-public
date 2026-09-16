"""Greenhouse board API adapter."""

import time

import requests

from src.models import Posting
from src.normalise import description_excerpt, employment_type, finalise, html_to_text

# content=true returns every description in the same request. It makes the
# payload several times larger, but the alternative is one request per posting.
URL = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true"


def fetch(source_id: str, board: str) -> list[Posting]:
    from src.fetchers import HEADERS, POLITE_DELAY, TIMEOUT

    r = requests.get(URL.format(board=board), headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    time.sleep(POLITE_DELAY)

    postings = []
    for job in r.json().get("jobs", []):
        title = job.get("title") or "(untitled)"
        location_raw = (job.get("location") or {}).get("name")
        postings.append(finalise(Posting(
            source_id=source_id,
            external_id=str(job.get("id")),
            title=title,
            url=job.get("absolute_url"),
            location_raw=location_raw,
            # Greenhouse exposes no employment-type field — title is all we get.
            employment_type=employment_type(title),
            description=description_excerpt(html_to_text(job.get("content"))),
        )))
    return postings
