"""Find which companies have a public ATS board, and write config/sources.yaml.

Greenhouse, Lever and Ashby board names are usually the company name lowercased
with spaces stripped, so instead of detecting the platform we just try the
endpoint: 200 means they are on it, 404 means they are not. A wrong guess costs
one failed request, which is why a long speculative list is fine.

    python tools/discover_boards.py            # probe everything, write sources.yaml
    python tools/discover_boards.py stripe imc  # probe specific names, print only

Not found here does NOT mean no internships — it means they are not on these
three platforms. Banks and large enterprises are usually on Workday or
SuccessFactors, which need their own adapters.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCES_PATH = ROOT / "config" / "sources.yaml"

# Singapore-relevant candidates. Wrong names are harmless — they just 404.
COMPANIES = [
    # Banks and finance with SG tech presence
    "dbs", "ocbc", "uob", "standardchartered", "hsbc", "citi", "jpmorgan",
    "goldmansachs", "morganstanley", "barclays", "ubs", "deutschebank",
    "macquarie", "nomura", "mufg", "bnpparibas", "creditagricole", "anz",
    "maybank", "cimb", "julius baer", "lombardodier", "pictet",

    # Trading, quant and market making
    "janestreet", "optiver", "imc", "jumptrading", "citadel", "citadelsecurities",
    "hudsonrivertrading", "drw", "sig", "susquehanna", "twosigma", "deshaw",
    "millennium", "point72", "balyasny", "qube", "tower research", "towerresearch",
    "flowtraders", "akunacapital", "belvederetrading", "vatic", "xtx",
    "gsa capital", "squarepoint", "wintermute", "vivienne court",
    # Board slugs that are not the company name. Guessing "optiver" or "drw"
    # finds nothing; these are the real ones, and they carry the Singapore and
    # London software internships those firms actually advertise.
    "optiverus", "drweng", "squarepointcapital",

    # Big tech with SG offices
    "google", "meta", "amazon", "microsoft", "apple", "netflix", "stripe",
    "airbnb", "uber", "lyft", "linkedin", "salesforce", "sap", "oracle",
    "ibm", "intel", "nvidia", "qualcomm", "dell", "vmware", "cisco", "adobe",

    # Regional tech and unicorns
    "grab", "sea", "shopee", "garena", "gojek", "tokopedia", "traveloka",
    "bytedance", "tiktok", "lazada", "carousell", "carro", "ninjavan",
    "propertyguru", "redoorz", "zilingo", "patsnap", "biofourmis", "advance",
    "aspire", "nium", "thunes", "matrixport", "coda", "codapayments",
    "circles", "circleslife", "glints", "endowus", "syfe", "stashaway",
    "funding societies", "fundingsocieties", "validus", "atome", "ryde",

    # Crypto and fintech (SG-heavy)
    "coinbase", "kraken", "binance", "crypto", "cryptocom", "amber", "ambergroup",
    "gemini", "chainalysis", "fireblocks", "ripple", "circle", "consensys",
    "revolut", "wise", "airwallex", "adyen", "checkout", "rapyd", "marqeta",
    "plaid", "block", "affirm", "klarna", "brex", "ramp",

    # Infra, data and AI
    "databricks", "snowflake", "confluent", "hashicorp", "datadog", "elastic",
    "mongodb", "redis", "cockroachlabs", "clickhouse", "dbtlabs", "fivetran",
    "airbyte", "temporal", "grafana", "gitlab", "github", "docker", "vercel",
    "netlify", "cloudflare", "fastly", "digitalocean", "linode",
    "openai", "anthropic", "scaleai", "huggingface", "cohere", "runwayml",
    "perplexityai", "mistral", "together", "modal", "replicate", "weightsandbiases",

    # Enterprise SaaS
    "atlassian", "figma", "canva", "notion", "linear", "asana", "monday",
    "slack", "zoom", "twilio", "segment", "amplitude", "mixpanel", "posthog",
    "intercom", "zendesk", "hubspot", "shopify", "squarespace", "wix",
    "doordash", "instacart", "robinhood", "chime", "sofi", "nubank",
    "palantir", "snyk", "1password", "okta", "auth0", "duo",

    # Consultancies and other
    "mckinsey", "bain", "bcg", "accenture", "thoughtworks", "deloitte",
    "shell", "exxonmobil", "dyson", "razer", "flex", "micron", "gic",
    "temasek", "singtel", "singaporeairlines", "sats", "psa", "keppel",
]

# Rough tags so a profile can select a subset via source_tags.
TAGS = {
    "trading": {"janestreet", "optiver", "imc", "jumptrading", "citadel",
                "hudsonrivertrading", "drw", "susquehanna", "twosigma", "deshaw",
                "millennium", "point72", "balyasny", "flowtraders", "akunacapital",
                "belvederetrading", "xtx", "gsacapital", "squarepoint", "wintermute",
                "optiverus", "drweng", "squarepointcapital"},
    "fintech": {"coinbase", "kraken", "binance", "crypto", "amber", "ambergroup",
                "gemini", "chainalysis", "fireblocks", "ripple", "circle",
                "consensys", "revolut", "wise", "airwallex", "adyen", "marqeta",
                "plaid", "block", "affirm", "klarna", "brex", "ramp", "nium",
                "thunes", "aspire", "endowus", "syfe", "stashaway", "nubank",
                "robinhood", "chime", "sofi"},
    "infra": {"databricks", "snowflake", "confluent", "hashicorp", "datadog",
              "elastic", "mongodb", "redis", "cockroachlabs", "clickhouse",
              "fivetran", "airbyte", "temporal", "grafana", "gitlab", "docker",
              "vercel", "netlify", "cloudflare", "fastly", "modal"},
    "ai": {"openai", "anthropic", "scaleai", "huggingface", "cohere", "mistral",
           "modal", "perplexityai"},
    "bigtech": {"google", "meta", "amazon", "microsoft", "apple", "netflix",
                "stripe", "airbnb", "uber", "lyft", "linkedin", "salesforce"},
    "saas": {"atlassian", "figma", "canva", "notion", "linear", "asana", "twilio",
             "amplitude", "mixpanel", "posthog", "intercom", "squarespace",
             "instacart", "okta", "snyk", "1password"},
    "sea": {"grab", "sea", "shopee", "garena", "ninjavan", "patsnap",
            "biofourmis", "carousell", "propertyguru", "glints"},
}

# Workday tenants cannot be guessed: tenant, data-centre number and site name
# all vary, and the real site names here (DBS_Careers, UOBExternal, lateral-us,
# "2") match no pattern. They are found with tools/detect_ats.py, which reads
# the myworkdayjobs URL straight out of a careers page, then verified against
# the API before being listed here. Merged into sources.yaml, never probed.
EXTRA_SOURCES = [
    {"id": "sggov-workday", "name": "singapore public service",
     "platform": "workday", "board": "sggovterp/wd102/PublicServiceCareers",
     "tags": ["government", "sea"],
     # Every posting is in Singapore, but locations are internal site names
     # ("SLA-REVENUE HOUSE"), so nothing canonicalises without this.
     "default_location": "Singapore"},
    {"id": "dbs-workday", "name": "dbs", "platform": "workday",
     "board": "dbs/wd3/DBS_Careers", "tags": ["bank", "fintech", "sea"]},
    {"id": "ocbc-workday", "name": "ocbc", "platform": "workday",
     "board": "ocbc/wd102/External", "tags": ["bank", "fintech", "sea"]},
    {"id": "uob-workday", "name": "uob", "platform": "workday",
     "board": "uobgroup/wd3/UOBExternal", "tags": ["bank", "fintech", "sea"]},
    {"id": "morganstanley-workday", "name": "morganstanley",
     "platform": "workday", "board": "ms/wd5/External",
     "tags": ["bank", "trading"]},
    {"id": "citi-workday", "name": "citi", "platform": "workday",
     "board": "citi/wd5/2", "tags": ["bank", "fintech"]},
    {"id": "bankofamerica-workday", "name": "bankofamerica",
     "platform": "workday", "board": "ghr/wd1/lateral-us", "tags": ["bank"]},
    {"id": "micron-workday", "name": "micron", "platform": "workday",
     "board": "micron/wd1/External", "tags": ["sea"]},
    {"id": "salesforce-workday", "name": "salesforce", "platform": "workday",
     "board": "salesforce/wd12/External_Career_Site",
     "tags": ["bigtech", "saas"]},
    # Oracle Cloud: where the banks that are not on Workday live. JPMorgan runs
    # ~7000 postings here, including the SG/HK 2027 summer analyst programmes.
    {"id": "jpmorgan-oraclecloud", "name": "jpmorgan",
     "platform": "oraclecloud", "board": "jpmc.fa.oraclecloud.com/CX_1001",
     "tags": ["bank", "fintech"]},
]

PLATFORMS = {
    "greenhouse": "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs",
    # Deliberately unlimited: limit=1 proves existence but reports a count of 1,
    # which makes a 300-posting board lose the pick_best_board comparison.
    "lever": "https://api.lever.co/v0/postings/{slug}?mode=json",
    "ashby": "https://api.ashbyhq.com/posting-api/job-board/{slug}",
    # Answers 200 for ANY slug, so only a non-zero count proves this board
    # exists. The `count > 0` filter below is what makes it safe to probe.
    "smartrecruiters": "https://api.smartrecruiters.com/v1/companies/{slug}/postings?limit=1",
}

HEADERS = {"User-Agent": "board-finder (personal job search script)"}
TIMEOUT = 12


def slugs(name: str) -> list[str]:
    base = name.strip().lower()
    return list(dict.fromkeys(v for v in
                              [base.replace(" ", ""), base.replace(" ", "-")] if v))


def tags_for(slug: str) -> list[str]:
    return sorted(tag for tag, members in TAGS.items() if slug in members)


def pick_best_board(hits):
    """One board per company: keep the fullest, drop the rest.

    A company on two platforms is usually a migration left half-finished — Wise
    had 21 postings on Greenhouse and 433 on SmartRecruiters. Keeping both would
    also notify the same job twice, since a different source_id makes it a
    different row.

    This is only sound because every platform is probed for its true count; an
    existence-only probe would make a large board look like a one-posting stub.
    """
    by_company: dict[str, list] = {}
    for hit in hits:
        by_company.setdefault(hit[0], []).append(hit)

    kept, dropped = [], []
    for company, boards in by_company.items():
        if len(boards) == 1:
            kept.append(boards[0])
            continue
        best = max(boards, key=lambda h: h[3])
        kept.append(best)
        for other in boards:
            if other is not best:
                dropped.append((*other, best[1], best[3]))
    return kept, dropped


def probe_one(args):
    name, slug, platform, url = args
    try:
        r = requests.get(url.format(slug=slug), headers=HEADERS, timeout=TIMEOUT)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None

    if isinstance(data, dict):
        # totalFound (SmartRecruiters) is the true size; the others return the
        # whole list, so its length is the count.
        count = data.get("totalFound")
        if count is None:
            count = len(data.get("jobs") or data.get("results") or [])
    elif isinstance(data, list):
        count = len(data)
    else:
        count = 0
    return (name, platform, slug, count)


def main() -> int:
    names = sys.argv[1:] or COMPANIES
    write = not sys.argv[1:]

    jobs = [(name, slug, platform, url)
            for name in names
            for slug in slugs(name)
            for platform, url in PLATFORMS.items()]
    print(f"testing {len(jobs)} combinations across {len(names)} companies\n")

    hits = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for result in pool.map(probe_one, jobs):
            if result:
                hits.append(result)
                name, platform, slug, count = result
                print(f"  {name:22} {platform:11} {slug:24} {count} postings")

    live = [h for h in hits if h[3] > 0]
    print(f"\n{len(hits)} boards found, {len(live)} non-empty.")

    live, dropped = pick_best_board(live)
    if dropped:
        print(f"\n{len(dropped)} stub/duplicate boards dropped "
              f"(same company listed on several platforms):")
        for name, platform, slug, count, kept_platform, kept_count in dropped:
            print(f"  {name:20} {platform:16} {count:>5} postings  "
                  f"-> keeping {kept_platform} ({kept_count})")

    if not write:
        return 0

    sources = [{"id": f"{slug}-{platform}", "name": name, "platform": platform,
                "board": slug, "tags": tags_for(slug)}
               for name, platform, slug, _ in sorted(live)]

    # Curated entries win: a probed board for the same company is a stub next to
    # these (Morgan Stanley's Workday board has 1315 postings; nothing else
    # lists them at all).
    curated_names = {s["name"] for s in EXTRA_SOURCES}
    shadowed = [s for s in sources if s["name"] in curated_names]
    sources = [s for s in sources if s["name"] not in curated_names] + EXTRA_SOURCES
    for s in shadowed:
        print(f"  superseded by curated workday source: {s['id']}")
    SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_PATH.write_text(
        yaml.safe_dump({"sources": sources}, sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    print(f"wrote {len(sources)} sources to {SOURCES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
