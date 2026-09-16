"""Lever postings API adapter."""

import time

import requests

from src.models import Posting
from src.normalise import description_excerpt, employment_type, finalise, html_to_text

URL = "https://api.lever.co/v0/postings/{board}?mode=json"


def fetch(source_id: str, board: str) -> list[Posting]:
    from src.fetchers import HEADERS, POLITE_DELAY, TIMEOUT

    r = requests.get(URL.format(board=board), headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    time.sleep(POLITE_DELAY)

    data = r.json()
    if not isinstance(data, list):  # error payloads come back as a dict
        return []

    postings = []
    for job in data:
        categories = job.get("categories") or {}
        title = job.get("text") or "(untitled)"
        # allLocations covers multi-city postings; fall back to the single field.
        all_locations = categories.get("allLocations") or []
        location_raw = "; ".join(all_locations) or categories.get("location")
        # Lever returns the full description in the list response, at no extra
        # request cost. `additionalPlain` usually carries the legal/visa boilerplate.
        body = " ".join(filter(None, [
            job.get("descriptionPlain"),
            job.get("additionalPlain"),
        ])) or html_to_text(job.get("description"))

        postings.append(finalise(Posting(
            source_id=source_id,
            external_id=str(job.get("id")),
            title=title,
            url=job.get("hostedUrl"),
            location_raw=location_raw,
            employment_type=employment_type(title, categories.get("commitment")),
            remote=(job.get("workplaceType") or "").lower() == "remote",
            description=description_excerpt(body),
        )))
    return postings
