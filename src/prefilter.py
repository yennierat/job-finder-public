"""Compiles a JobProfile into deterministic keep/drop rules.

Deliberately loose. People state preferences far more narrowly than they would
actually accept, so the prefilter only drops what is clearly wrong and leaves
nuance to the classifier. Over-fetching is cheap; never seeing a posting is not.
"""

import re

from src.models import JobProfile, Posting
from src.normalise import expand_region


class Prefilter:
    def __init__(self, profile: JobProfile):
        self.types = set(profile.employment_types)
        # Regions are expanded to member cities: canonical_location only ever
        # yields a city, so an unexpanded "Europe" would match nothing.
        self.locations = {city
                          for loc in profile.locations if loc.strip()
                          for city in expand_region(loc)}
        self.remote_ok = profile.remote_ok
        self.exclude = [re.compile(rf"\b{re.escape(k)}\b", re.I)
                        for k in profile.exclude_keywords if k.strip()]
        self.must_have = [re.compile(rf"\b{re.escape(k)}\b", re.I)
                          for k in profile.must_have_keywords if k.strip()]

    def keep(self, p: Posting) -> bool:
        if self.exclude and any(rx.search(p.title) for rx in self.exclude):
            return False

        # must_have is an OR, not an AND: any one hit is enough. Requiring all of
        # them would drop postings that simply phrase the role differently.
        if self.must_have and not any(rx.search(p.title) for rx in self.must_have):
            return False

        if self.types and p.employment_type and p.employment_type not in self.types:
            return False

        if self.locations:
            if p.remote and self.remote_ok:
                return True
            # An unrecognised location (canonicalisation returned None) is kept
            # on purpose — unknown is not the same as "somewhere else".
            if p.location is not None and p.location not in self.locations:
                return False

        return True

    def apply(self, postings: list[Posting]) -> list[Posting]:
        return [p for p in postings if self.keep(p)]
