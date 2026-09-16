"""Fetcher registry — one adapter per ATS platform."""

from src.fetchers import (amazonjobs, ashby, greenhouse, lever, oraclecloud,
                          smartrecruiters, workday)

REGISTRY = {
    "greenhouse": greenhouse.fetch,
    "lever": lever.fetch,
    "ashby": ashby.fetch,
    "smartrecruiters": smartrecruiters.fetch,
    "workday": workday.fetch,
    "oraclecloud": oraclecloud.fetch,
    "amazonjobs": amazonjobs.fetch,
}

# Platforms whose list response carries no description, and which therefore need
# a second request per posting to be judged on anything but a job title. Only
# these are listed: the rest already return the advert text in bulk, and paying
# for a detail call there would be a request per posting for nothing.
DETAIL_REGISTRY = {
    "workday": workday.detail,
    "smartrecruiters": smartrecruiters.detail,
}

# Undocumented internal endpoints: identify honestly and stay slow.
HEADERS = {"User-Agent": "internship-monitor (personal job search; contact via repo)"}
TIMEOUT = 20
POLITE_DELAY = 0.3


def fetch(platform: str, source_id: str, board: str):
    try:
        adapter = REGISTRY[platform]
    except KeyError:
        raise ValueError(f"unknown platform: {platform}")
    return adapter(source_id, board)


def needs_detail(platform: str) -> bool:
    return platform in DETAIL_REGISTRY


def detail(platform: str, source_id: str, board: str, posting):
    """(description, deadline) for one posting, or (None, None) if unavailable."""
    adapter = DETAIL_REGISTRY.get(platform)
    if adapter is None:
        return None, None
    return adapter(source_id, board, posting)
