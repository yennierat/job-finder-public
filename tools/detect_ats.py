"""Detect which ATS a company's careers page is built on.

Guessing Workday tenant names is unreliable — a miss proves nothing. Reading
the careers page is definitive: if it embeds a myworkdayjobs.com URL, the tenant
and site name are right there in the link.

    python tools/detect_ats.py            # check the built-in list
    python tools/detect_ats.py https://careers.example.com/
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; job-hunt personal job search)"}
TIMEOUT = 25

MARKERS = {
    "workday": r"myworkdayjobs\.com",
    "taleo": r"taleo\.net",
    "successfactors": r"successfactors\.(?:com|eu)|jobs\.sap\.com",
    "avature": r"avature\.net",
    "icims": r"icims\.com",
    "eightfold": r"eightfold\.ai",
    "phenom": r"phenompeople\.com",
    "oracle_cloud": r"oraclecloud\.com",
    "greenhouse": r"greenhouse\.io",
    "lever": r"jobs\.lever\.co|api\.lever\.co",
    "ashby": r"ashbyhq\.com",
    "smartrecruiters": r"smartrecruiters\.com",
    "brassring": r"brassring\.com",
}

# The path may or may not start with a locale ("/en-US/Careers" vs "/Careers"),
# so capture two segments and let the caller try both readings — a locale
# mistaken for the site name yields "Job_Posting_Site_ID=en-us not found".
WD_URL_RE = re.compile(
    r"https?://([\w-]+)\.(wd\d+)\.myworkdayjobs\.com/([\w-]+)(?:/([\w-]+))?")

CAREERS_PAGES = {
    "jpmorgan": "https://careers.jpmorgan.com/global/en/home",
    "morganstanley": "https://www.morganstanley.com/careers",
    "ubs": "https://www.ubs.com/global/en/careers.html",
    "bankofamerica": "https://careers.bankofamerica.com/en-us",
    "goldmansachs": "https://www.goldmansachs.com/careers",
    "citi": "https://jobs.citi.com/",
    "hsbc": "https://www.hsbc.com/careers",
    "standardchartered": "https://www.sc.com/en/careers/",
    "dbs": "https://www.dbs.com/careers/default.page",
    "ocbc": "https://www.ocbc.com/group/careers/index.page",
    "uob": "https://www.uobgroup.com/careers/index.page",
    "barclays": "https://home.barclays/careers/",
    "deutschebank": "https://careers.db.com/index_e.htm",
    "bnpparibas": "https://group.bnpparibas/en/careers",
    "nomura": "https://www.nomura.com/careers/",
    "macquarie": "https://www.macquarie.com/us/en/careers.html",
    "wellsfargo": "https://www.wellsfargojobs.com/",
    "jefferies": "https://www.jefferies.com/careers/",
    "schroders": "https://www.schroders.com/en/global/individual/careers/",
    "aberdeen": "https://www.aberdeenplc.com/careers",
}


def detect(item):
    name, url = item
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    except requests.RequestException as e:
        return name, url, f"ERROR {type(e).__name__}", []
    if r.status_code != 200:
        return name, url, f"HTTP {r.status_code}", []

    found = [ats for ats, pat in MARKERS.items() if re.search(pat, r.text, re.I)]
    tenants = set()
    for t, wd, first, second in WD_URL_RE.findall(r.text):
        tenants.add(f"{t}/{wd}/{first}")
        if second:
            tenants.add(f"{t}/{wd}/{second}")
    return name, r.url, ", ".join(found) or "none detected", sorted(tenants)


def main() -> int:
    if len(sys.argv) > 1:
        pages = {u: u for u in sys.argv[1:]}
    else:
        pages = CAREERS_PAGES

    with ThreadPoolExecutor(max_workers=6) as pool:
        for name, url, ats, tenants in pool.map(detect, pages.items()):
            print(f"{name:20} {ats}")
            for t in tenants:
                print(f"{'':20}   workday board: {t}")
    print("\nA marker on the landing page is a strong hint, not proof — some "
          "sites load the ATS only on an inner page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
