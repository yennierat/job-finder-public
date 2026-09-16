"""Contract tests: every adapter, replayed against a recorded real response.

Fixtures in tests/fixtures/ were captured from the live APIs. If a vendor
changes its JSON shape the fixture goes stale, but these still guard the thing
that actually breaks in practice — a refactor silently dropping a field, so a
run returns postings with no url, no location or the wrong employment type.

No network: requests.get/post are replaced per adapter module.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fetchers import (REGISTRY, amazonjobs, ashby,  # noqa: E402
                          greenhouse, lever, oraclecloud, smartrecruiters,
                          workday)
from src.normalise import CITY_PATTERNS  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
failures = []


def check(name, got, want):
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


def ok(name, cond):
    if not cond:
        failures.append(f"{name}: expected true")


def load(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def replay(module, fixture, source_id, board, verb="get"):
    """Run one adapter against its fixture, serving the payload once.

    The second call returns an empty page so paginating adapters terminate
    instead of looping on the same fixture.
    """
    payload = load(fixture)
    empty = {"jobs": [], "content": [], "jobPostings": [],
             "items": [], "hits": 0, "total": 0, "totalFound": 0}
    calls = {"n": 0}

    def fake(*a, **kw):
        calls["n"] += 1
        return FakeResponse(payload if calls["n"] == 1 else empty)

    def forbidden(*a, **kw):
        raise AssertionError(
            f"{module.__name__} made an unexpected {verb}-mismatched request; "
            "no test may reach the network")

    real_get, real_post = module.requests.get, module.requests.post
    real_sleep = module.time.sleep
    # Both verbs are replaced: requests is one shared module, so leaving the
    # unused verb live would let a stray call reach a real ATS from CI.
    module.requests.get = fake if verb == "get" else forbidden
    module.requests.post = fake if verb == "post" else forbidden
    module.time.sleep = lambda *_: None
    try:
        return module.fetch(source_id, board)
    finally:
        module.requests.get, module.requests.post = real_get, real_post
        module.time.sleep = real_sleep


CASES = [
    ("greenhouse", greenhouse, "greenhouse", "imc-greenhouse", "imc", "get"),
    ("lever", lever, "lever", "palantir-lever", "palantir", "get"),
    ("ashby", ashby, "ashby", "airwallex-ashby", "airwallex", "get"),
    ("smartrecruiters", smartrecruiters, "smartrecruiters",
     "grab-smartrecruiters", "grab", "get"),
    ("workday", workday, "workday", "ms-workday", "ms/wd5/External", "post"),
    ("oraclecloud", oraclecloud, "oraclecloud", "jpm-oraclecloud",
     "jpmc.fa.oraclecloud.com/CX_1001", "get"),
    ("amazonjobs", amazonjobs, "amazonjobs", "amazon-amazonjobs",
     "internship", "get"),
]

CANONICAL = set(CITY_PATTERNS)

for name, module, fixture, source_id, board, verb in CASES:
    postings = replay(module, fixture, source_id, board, verb)

    ok(f"{name}: returns postings", len(postings) > 0)
    if not postings:
        continue

    ids = [p.external_id for p in postings]
    # UOB puts the office location in the field that usually holds a requisition
    # number; duplicate ids meant 96% of one board was silently discarded.
    check(f"{name}: ids unique", len(set(ids)), len(ids))
    ok(f"{name}: ids non-empty", all(i and i.strip() for i in ids))
    ok(f"{name}: source_id propagated",
       all(p.source_id == source_id for p in postings))
    ok(f"{name}: titles non-empty", all(p.title and p.title.strip() for p in postings))
    ok(f"{name}: urls look like links",
       all(p.url is None or p.url.startswith("http") for p in postings))
    # Asserting membership of the EmploymentType literal is tautological —
    # pydantic already enforces it. Require instead that detection actually ran
    # and reached a verdict for every posting.
    ok(f"{name}: employment_type populated",
       all(p.employment_type is not None for p in postings))
    # finalise() runs inside every adapter, so this must always be populated.
    ok(f"{name}: content_hash set", all(p.content_hash for p in postings))
    # Likewise: `isinstance(str)` is tautological. Require that a resolved
    # location is genuinely canonical — a raw ATS string such as
    # "LONDON, LONDON, United Kingdom" must never survive into `location`.
    resolved = [p for p in postings if p.location]
    ok(f"{name}: locations canonicalised",
       all(p.location in CANONICAL for p in resolved))
    ok(f"{name}: location_raw preserved",
       all(p.location_raw is not None for p in resolved))

# Adapters that get descriptions for free must actually extract them; the ones
# that would need a request per posting must not pretend to.
WITH_DESCRIPTIONS = {"greenhouse": greenhouse, "lever": lever, "ashby": ashby,
                     "oraclecloud": oraclecloud, "amazonjobs": amazonjobs}
for name, module in WITH_DESCRIPTIONS.items():
    case = next(c for c in CASES if c[0] == name)
    postings = replay(module, case[2], case[3], case[4], case[5])
    ok(f"{name}: supplies descriptions", any(p.description for p in postings))
    ok(f"{name}: no raw html in description",
       all("<p>" not in (p.description or "") for p in postings))

# Every registered platform must have a case here, so a new adapter cannot be
# added without a contract test.
check("all adapters covered", sorted(REGISTRY), sorted(c[0] for c in CASES))

# Board strings that adapters parse must fail loudly, not silently.
for module, bad in ((workday, "ms/wd5"), (oraclecloud, "hostonly")):
    raised = False
    try:
        module.parse_board(bad)
    except ValueError:
        raised = True
    check(f"{module.__name__}: rejects malformed board {bad!r}", raised, True)

# --- detail endpoints -----------------------------------------------------
# Only the platforms whose list response carries no description have one, and
# their output must be capped the same way the bulk adapters cap theirs.

from src.fetchers import DETAIL_REGISTRY  # noqa: E402
from src.models import Posting  # noqa: E402
from src.normalise import description_excerpt  # noqa: E402

check("only text-less platforms have a detail fetcher",
      sorted(DETAIL_REGISTRY), ["smartrecruiters", "workday"])


class DetailResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status
        self.ok = status == 200

    def json(self):
        return self._payload


LONG = "<p>" + ("Responsibilities and requirements. " * 300) + "</p>"


def replay_detail(module, payload, board, external_id, status=200):
    real_get, real_sleep = module.requests.get, module.time.sleep
    module.requests.get = lambda *a, **kw: DetailResponse(payload, status)
    module.time.sleep = lambda *_: None
    try:
        return module.detail("src", board,
                             Posting(source_id="src", external_id=external_id,
                                     title="T"))
    finally:
        module.requests.get, module.time.sleep = real_get, real_sleep


desc, deadline = replay_detail(
    workday, {"jobPostingInfo": {"jobDescription": LONG,
                                 "endDate": "2026-09-30"}},
    "ms/wd5/External", "/job/Singapore/Engineer_JR1")
ok("workday detail returns text", desc and "Responsibilities" in desc)
ok("workday detail strips html", "<p>" not in (desc or ""))
check("workday detail reads endDate", deadline, "2026-09-30")

# Detail returns the FULL advert, so enrich() must excerpt it. If it did not,
# six postings would carry ~34,000 characters into one prompt and bury each
# other. This asserts the raw text really is the size that makes that matter.
ok("detail text is genuinely long", len(desc) > 3000)
ok("excerpting it brings it into line",
   len(description_excerpt(desc)) < 1200)

# A tenant that refuses (UOB and Citi both 403) yields nothing, never an error.
check("workday 403 yields nothing",
      replay_detail(workday, {}, "ms/wd5/External", "/job/x", status=403),
      (None, None))
# An id that is not an externalPath cannot address the detail endpoint.
check("non-path id is not requested",
      replay_detail(workday, {"jobPostingInfo": {"jobDescription": "x"}},
                    "ms/wd5/External", "REQ-123"),
      (None, None))

sr_desc, sr_deadline = replay_detail(
    smartrecruiters,
    {"jobAd": {"sections": {
        "companyDescription": {"text": "We are a big company. " * 40},
        "jobDescription": {"text": "You will write software."},
        "qualifications": {"text": "Python required."},
        "additionalInformation": {"text": "Equal opportunity."}}}},
    "grab", "123")
ok("smartrecruiters joins the useful sections",
   "write software" in sr_desc and "Python required" in sr_desc)
# companyDescription is identical boilerplate on every posting; including it
# would crowd out the part that distinguishes one job from another.
ok("smartrecruiters drops company boilerplate", "big company" not in sr_desc)
check("smartrecruiters exposes no closing date", sr_deadline, None)

check("workday board parses", workday.parse_board("ms/wd5/External"),
      ("ms", "wd5", "External"))
check("oracle board parses", oraclecloud.parse_board("host.example.com/CX_1"),
      ("host.example.com", "CX_1"))


if failures:
    print(f"{len(failures)} FAILURES:")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("fetcher tests passed")
