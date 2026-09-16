"""Audit source coverage, and probe whether missing companies are on Workday.

Two questions, because they have different answers:

  1. What does sources.yaml actually cover, and which platforms have no adapter?
     Answered from local files — cheap and certain.

  2. Are the companies we're missing actually Workday tenants?
     Answered by probing Workday's public CXS endpoint. A HIT is proof. A miss
     proves nothing: tenant and site names are guesses, so a company can be on
     Workday under a name this script never tried.

    python tools/audit_sources.py            # local audit only
    python tools/audit_sources.py --probe    # also probe Workday tenancy
"""

import collections
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fetchers import REGISTRY  # noqa: E402

SOURCES_PATH = ROOT / "config" / "sources.yaml"

# Companies the current corpus is missing, and where they plausibly live.
MISSING = [
    "dbs", "ocbc", "uob", "standardchartered", "hsbc", "citi", "jpmorgan",
    "goldmansachs", "morganstanley", "barclays", "ubs", "macquarie", "nomura",
    "grab", "sea", "shopee", "bytedance", "tiktok", "google", "meta", "amazon",
    "microsoft", "apple", "nvidia", "salesforce", "atlassian", "canva",
    "mckinsey", "bain", "accenture", "deloitte", "micron", "singtel",
]

# Workday's public job-search API. Tenant, data-centre number and site name all
# vary per customer, so this is a guess matrix rather than a lookup.
WD_URL = "https://{tenant}.wd{n}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"
WD_NUMBERS = [1, 2, 3, 5, 12, 103]
WD_SITES = ["External", "careers", "Careers", "External_Career_Site",
            "en-US/External", "Externalcareers"]

HEADERS = {
    "User-Agent": "job-hunt source audit (personal job search)",
    "Content-Type": "application/json",
    "Accept": "application/json",
}
BODY = {"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": ""}
TIMEOUT = 10


def local_audit() -> set[str]:
    data = yaml.safe_load(SOURCES_PATH.read_text(encoding="utf-8"))
    sources = data.get("sources", [])
    by_platform = collections.Counter(s["platform"] for s in sources)

    print(f"sources.yaml: {len(sources)} sources")
    for platform, n in by_platform.most_common():
        adapter = "ok" if platform in REGISTRY else "NO ADAPTER"
        print(f"  {platform:12} {n:>4} sources   [{adapter}]")

    unused = sorted(set(REGISTRY) - set(by_platform))
    if unused:
        print(f"  adapters with no sources: {', '.join(unused)}")

    boards = {s["board"] for s in sources}
    absent = [c for c in MISSING if c.replace(" ", "") not in boards]
    print(f"\n{len(absent)} of {len(MISSING)} checked companies are absent "
          "from sources.yaml")
    return set(absent)


def probe_workday(tenant: str):
    """Return (url, count) on the first hit, else None."""
    for n in WD_NUMBERS:
        for site in WD_SITES:
            url = WD_URL.format(tenant=tenant, n=n, site=site)
            try:
                r = requests.post(url, headers=HEADERS, json=BODY, timeout=TIMEOUT)
            except requests.RequestException:
                continue
            if r.status_code != 200:
                continue
            try:
                body = r.json()
            except ValueError:
                continue
            if "jobPostings" in body:
                return url, body.get("total", len(body["jobPostings"]))
    return None


def main() -> int:
    absent = local_audit()

    if "--probe" not in sys.argv:
        print("\n(run with --probe to test Workday tenancy for the absent companies)")
        return 0

    print(f"\nprobing Workday for {len(absent)} companies "
          f"({len(WD_NUMBERS) * len(WD_SITES)} guesses each)...\n")

    hits = []
    from concurrent.futures import ThreadPoolExecutor

    # Guesses run sequentially within a tenant, tenants in parallel.
    with ThreadPoolExecutor(max_workers=6) as pool:
        for tenant, result in zip(sorted(absent),
                                  pool.map(probe_workday, sorted(absent))):
            if result:
                url, count = result
                hits.append((tenant, url, count))
                print(f"  HIT   {tenant:18} {count:>6} postings   {url}")

    print(f"\n{len(hits)} of {len(absent)} confirmed on Workday.")
    print("A miss is not proof of absence — tenant/site names are guessed here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
