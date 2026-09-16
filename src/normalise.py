"""Canonicalise ATS location strings and derive employment type.

ATS location strings are chaos: "SG", "Singapore, Singapore",
"Asia Pacific - Singapore", "SG - Singapore". Normalise once here, at the
boundary, so every downstream stage deals in one vocabulary.
"""

import hashlib
import html
import re
from datetime import date, timedelta

from src.models import Posting

# Canonical name -> patterns that should map onto it. Order matters only in that
# the first canonical match wins, so keep entries mutually exclusive.
CITY_PATTERNS: dict[str, list[str]] = {
    # Banks label offices by building or planning region rather than by city.
    # "Central Region (City Area)" is how UOB tags Singapore; "One Island East"
    # is a Taikoo Place tower in Hong Kong. Both are matched narrowly, since
    # "central region" alone occurs in several other countries.
    "Singapore": [r"\bsingapore\b", r"\bsg\b", r"\bsin\b",
                  r"central region \(city area\)"],
    "Hong Kong": [r"\bhong ?kong\b", r"\bhk\b", r"\bhkg\b",
                  r"\bone island east\b"],
    "Tokyo": [r"\btokyo\b"],
    "Sydney": [r"\bsydney\b"],
    "London": [r"\blondon\b"],
    "New York": [r"\bnew york\b", r"\bnyc\b", r"\bny\b"],
    "San Francisco": [r"\bsan francisco\b", r"\bsf\b", r"\bbay area\b"],
    "Seattle": [r"\bseattle\b"],
    "Chicago": [r"\bchicago\b"],
    "Amsterdam": [r"\bamsterdam\b"],
    "Dublin": [r"\bdublin\b"],
    "Vienna": [r"\bvienna\b", r"\bwien\b"],
    "Zurich": [r"\bzurich\b", r"\bzürich\b"],
    "Geneva": [r"\bgeneva\b"],
    "Munich": [r"\bmunich\b", r"\bmünchen\b"],
    "Madrid": [r"\bmadrid\b"],
    "Barcelona": [r"\bbarcelona\b"],
    "Stockholm": [r"\bstockholm\b"],
    "Copenhagen": [r"\bcopenhagen\b"],
    "Oslo": [r"\boslo\b"],
    "Helsinki": [r"\bhelsinki\b"],
    "Warsaw": [r"\bwarsaw\b"],
    "Krakow": [r"\bkrakow\b", r"\bkraków\b"],
    "Prague": [r"\bprague\b"],
    "Budapest": [r"\bbudapest\b"],
    "Lisbon": [r"\blisbon\b"],
    "Milan": [r"\bmilan\b"],
    "Brussels": [r"\bbrussels\b"],
    "Bucharest": [r"\bbucharest\b"],
    "Edinburgh": [r"\bedinburgh\b"],
    "Manchester": [r"\bmanchester\b"],
    "Cambridge": [r"\bcambridge\b"],
    "Bangalore": [r"\bbangalore\b", r"\bbengaluru\b"],
    "Shanghai": [r"\bshanghai\b"],
    "Beijing": [r"\bbeijing\b"],
    "Seoul": [r"\bseoul\b"],
    "Kuala Lumpur": [r"\bkuala lumpur\b", r"\bkl\b"],
    "Jakarta": [r"\bjakarta\b"],
    "Bangkok": [r"\bbangkok\b"],
    "Manila": [r"\bmanila\b"],
    "Ho Chi Minh City": [r"\bho chi minh\b", r"\bsaigon\b"],
    "Taipei": [r"\btaipei\b"],
    "Mumbai": [r"\bmumbai\b"],
    "Berlin": [r"\bberlin\b"],
    "Paris": [r"\bparis\b"],
    "Toronto": [r"\btoronto\b"],
    "Austin": [r"\baustin\b"],
    "Boston": [r"\bboston\b"],
    "Los Angeles": [r"\blos angeles\b", r"\bla\b"],
    "Dubai": [r"\bdubai\b"],
    "Tel Aviv": [r"\btel aviv\b"],
}

_COMPILED = {city: [re.compile(p, re.I) for p in pats]
             for city, pats in CITY_PATTERNS.items()}

# People name regions ("Europe", "APAC"), but canonical_location only ever
# returns a city. Without this expansion a profile asking for Europe would match
# nothing, because "London" is not the string "Europe".
REGIONS: dict[str, set[str]] = {
    "Europe": {"London", "Amsterdam", "Dublin", "Berlin", "Paris", "Vienna",
               "Zurich", "Geneva", "Munich", "Madrid", "Barcelona", "Stockholm",
               "Copenhagen", "Oslo", "Helsinki", "Warsaw", "Krakow", "Prague",
               "Budapest", "Lisbon", "Milan", "Brussels", "Bucharest",
               "Edinburgh", "Manchester", "Cambridge"},
    "APAC": {"Singapore", "Hong Kong", "Tokyo", "Sydney", "Seoul", "Shanghai",
             "Beijing", "Taipei", "Kuala Lumpur", "Jakarta", "Bangkok", "Manila",
             "Ho Chi Minh City", "Bangalore", "Mumbai"},
    "United States": {"New York", "San Francisco", "Seattle", "Chicago", "Austin",
                      "Boston", "Los Angeles"},
    "Middle East": {"Dubai", "Tel Aviv"},
}
# Aliases people actually type.
REGIONS["EU"] = REGIONS["Europe"]
REGIONS["Asia"] = REGIONS["APAC"]
REGIONS["Asia Pacific"] = REGIONS["APAC"]
REGIONS["Southeast Asia"] = {"Singapore", "Kuala Lumpur", "Jakarta", "Bangkok",
                             "Manila", "Ho Chi Minh City"}
REGIONS["US"] = REGIONS["United States"]
REGIONS["USA"] = REGIONS["United States"]
REGIONS["North America"] = REGIONS["United States"] | {"Toronto"}

_REGION_LOOKUP = {name.lower(): cities for name, cities in REGIONS.items()}


def expand_region(name: str) -> set[str]:
    """Region name -> member cities; a plain city returns just itself."""
    return _REGION_LOOKUP.get(name.strip().lower(), {name})

REMOTE_RE = re.compile(r"\bremote\b|\bwork from home\b|\banywhere\b", re.I)

# Banks and professional-services firms almost never say "intern". A summer
# internship at Citi or Morgan Stanley is a "Summer Analyst". Missing these
# silently drops an entire industry's early-careers hiring.
#
# Deliberately excluded: spring weeks, insight programmes and vacation schemes.
# Those are one-week tasters, not internships.
INTERN_RE = re.compile(
    r"\bintern(?:ship)?\b|\bco-?op\b|"
    r"\bsummer (?:analyst|associate|scholar)\b|"
    r"\boff[\s-]?cycle\b|"
    r"\b(?:industrial |year[\s-]?long )?placement(?: year)?\b",
    re.I)
GRAD_RE = re.compile(r"\bgraduate (?:programme|program|scheme)\b|\bnew ?grad\b"
                     r"|\bcampus\b|\buniversity ?grad", re.I)
PART_TIME_RE = re.compile(r"\bpart[\s-]?time\b", re.I)
CONTRACT_RE = re.compile(
    r"\bcontract(?:or)?\b|\btemp(?:orary)?\b|\bfixed[\s-]?term\b", re.I)


def canonical_location(raw: str | None) -> str | None:
    """Map a raw ATS location string onto a canonical city name.

    Returns None when nothing matches — callers must treat that as "unknown",
    never as "no match", so unrecognised locations stay visible to the LLM
    rather than being silently filtered out.
    """
    if not raw:
        return None
    for city, patterns in _COMPILED.items():
        if any(p.search(raw) for p in patterns):
            return city
    return None


def is_remote(raw: str | None) -> bool:
    return bool(raw and REMOTE_RE.search(raw))


def employment_type(title: str, structured: str | None = None) -> str:
    """Prefer the ATS's own structured field; fall back to the title.

    Greenhouse exposes no employment-type field at all, so title-grep is the
    only signal there.
    """
    text = structured or ""
    if INTERN_RE.search(text) or INTERN_RE.search(title):
        return "internship"
    if GRAD_RE.search(title):
        return "graduate_programme"
    if PART_TIME_RE.search(text) or PART_TIME_RE.search(title):
        return "contract"  # part-time is rare and unreliable; treat as non-FT
    if CONTRACT_RE.search(text) or CONTRACT_RE.search(title):
        return "contract"
    return "full_time"


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]*\n\s*|\s{2,}")

# Visa and right-to-work language, which almost always sits near the END of a
# description — after the role, requirements and benefits. A naive head-truncation
# therefore cuts off exactly the part we most want the classifier to see.
_SENTENCE_RE = re.compile(r"[^.\n]+[.\n]?")

# Unambiguous on their own.
_AUTHZ_STRONG = re.compile(
    r"\b(?:sponsor\w*|work(?:ing)? (?:authoriz|authoris|permit|right)\w*|"
    r"right to work|eligib\w+ to work|employment pass|work pass|"
    r"immigration status|permanent resident\w*|"
    # Both word orders: "work authorization" and "authorized to work".
    r"authoriz\w* to work|authoris\w* to work|legally (?:authoriz|authoris|entitled)\w*"
    r")\b", re.I)

# Ambiguous alone — "Visa" is a payment network and a common investor name, and
# "citizenship" appears in diversity boilerplate. Require a work-context word in
# the same sentence before treating these as authorization language.
_AUTHZ_WEAK = re.compile(
    r"\b(?:visas?|citizens?h?i?p?|immigration|nationality)\b", re.I)
_AUTHZ_CONTEXT = re.compile(
    r"\b(?:work|employ\w*|hir\w+|permit|require\w*|eligib\w*|authoriz\w*|"
    r"authoris\w*|sponsor\w*|status|relocat\w*|legally|entitled)\b", re.I)


# Equal-opportunity boilerplate lists protected characteristics — and
# "citizenship" and "national origin" are among them, so it trips the weak
# patterns on nearly every posting. Recognised by the pile-up of such terms.
_EEO_TERMS_RE = re.compile(
    r"\b(?:race|colou?r|religion|creed|sexual orientation|gender identity|"
    r"gender expression|ancestry|marital status|family status|disabilit\w+|"
    r"veteran|pregnan\w+|age|genetic\w*|protected (?:class|characteristic|status)|"
    r"equal opportunit\w+|discriminat\w+)\b", re.I)


def _is_authz_sentence(sentence: str) -> bool:
    if _AUTHZ_STRONG.search(sentence):
        return True
    if not (_AUTHZ_WEAK.search(sentence) and _AUTHZ_CONTEXT.search(sentence)):
        return False
    # Two or more protected characteristics means this is a non-discrimination
    # statement that happens to mention citizenship, not a visa requirement.
    return len(_EEO_TERMS_RE.findall(sentence)) < 2

# --- application deadlines ------------------------------------------------
# Extracted by rule, never by the model. A date is the one field where a
# hallucination is actively dangerous: a wrong visa summary is checked when you
# read the advert, but a wrong deadline is trusted and acted on, and being told
# "closes in 3 weeks" about something closing tomorrow costs the application.
# A regex either matches text that is really there or finds nothing.

_DEADLINE_TRIGGER = re.compile(
    r"(?:application|applications|apply|submission|submissions|registration)"
    r"[^.\n]{0,40}?"
    r"(?:deadline|close[sd]?|closing|due|by|before|end[s]?)"
    r"|(?:deadline|closing date|last day to apply|apply before|apply by)",
    re.I)

_MONTHS = ("january february march april may june july august september "
           "october november december").split()
_MONTH_NUM = {m[:3]: i for i, m in enumerate(_MONTHS, 1)}

_DATE_PATTERNS = (
    # 27 September 2026 / 27 Sept 2026 / 27th Sep, 2026
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b"),
    # September 27, 2026 / Sep 27 2026
    re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"),
    # 2026-09-27
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
)


def _build_date(a: str, b: str, c: str, style: int) -> str | None:
    try:
        if style == 0:
            day, month, year = int(a), _MONTH_NUM.get(b[:3].lower()), int(c)
        elif style == 1:
            month, day, year = _MONTH_NUM.get(a[:3].lower()), int(b), int(c)
        else:
            year, month, day = int(a), int(b), int(c)
        if not month or not 1 <= day <= 31 or not 1 <= month <= 12:
            return None
        return date(year, month, day).isoformat()
    except (ValueError, TypeError):
        return None


def extract_deadline(text: str | None, today: date | None = None,
                     window: int = 80) -> tuple[str | None, str | None]:
    """Find an application deadline in advert text.

    Returns (iso_date, the sentence it came from) so every date can be checked
    against the source. A date is accepted only when it sits close after a
    phrase that means "deadline" — job adverts are full of other dates (start
    dates, founding years, programme dates), and taking any date at all would be
    worse than finding none.
    """
    if not text:
        return None, None
    today = today or date.today()

    for trigger in _DEADLINE_TRIGGER.finditer(text):
        window_text = text[trigger.start():trigger.end() + window]
        for style, pattern in enumerate(_DATE_PATTERNS):
            m = pattern.search(window_text)
            if not m:
                continue
            iso = _build_date(*m.groups(), style=style)
            if not iso:
                continue
            # Deadlines sit in the near future. A date years past, or a decade
            # out, means the trigger matched something that was not a deadline.
            found = date.fromisoformat(iso)
            if not (today - timedelta(days=30) <= found
                    <= today + timedelta(days=3 * 365)):
                continue
            sentence = _sentence_around(text, trigger.start())
            return iso, sentence
    return None, None


def _sentence_around(text: str, index: int, span: int = 160) -> str:
    start = max(0, text.rfind(".", 0, index) + 1)
    end = text.find(".", index)
    end = len(text) if end == -1 else end + 1
    return text[start:end].strip()[:span] or text[index:index + span].strip()


def days_until(iso: str | None, today: date | None = None) -> int | None:
    """Whole days until an ISO date; negative once it has passed."""
    if not iso:
        return None
    try:
        return (date.fromisoformat(iso) - (today or date.today())).days
    except (ValueError, TypeError):
        return None


def html_to_text(raw: str | None) -> str | None:
    """Strip markup to plain text.

    Unescapes before AND after stripping tags: Greenhouse double-escapes its
    content ("&lt;p&gt;" rather than "<p>"), so stripping first would leave the
    markup as literal text in the excerpt.
    """
    if not raw:
        return None
    text = html.unescape(raw)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    return text or None


# Sentences stating when a role runs, or how long for. These are rescued from
# beyond the head cutoff for the same reason the visa clauses are: they are
# disqualifying, and they are written at the END of an advert.
#
# Found the hard way. A Singapore data-platform internship matched at fit 78 on
# the strength of its opening paragraph, while the line "Able to commit full
# time from August 2026 to December 2026 ... for at least 4 to 6 months" sat at
# character 1,500 of a 1,699-character advert and was never sent to the model.
# For a candidate who can only work one summer, that single sentence decides it.
_TIMING_RE = re.compile(
    r"\b(?:able to commit|must be available|availability|commit(?:ment)? to|"
    r"full[\s-]?time from|duration of|for a (?:minimum|period) of|"
    r"minimum (?:of )?\d+\s*(?:to\s*\d+\s*)?month|"
    r"\d+\s*(?:to|-|–)\s*\d+\s*month|start(?:ing|s)? (?:date|in|on|from))\b",
    re.I)


def _is_timing_sentence(sentence: str) -> bool:
    return bool(_TIMING_RE.search(sentence))


def description_excerpt(description: str | None, head: int = 500,
                        max_authz: int = 400,
                        max_timing: int = 300) -> str | None:
    """Head of the description, plus the sentences that can disqualify it.

    Sending whole descriptions would blow the context budget on a batch of six.
    The head carries what the role actually is; the appended clauses carry the
    two things that override it — visa terms and dates — both of which are
    conventionally written last and would otherwise be truncated away.
    """
    if not description:
        return None
    text = description.strip()
    excerpt = text[:head]

    tail = text[head:]
    authz: list[str] = []
    timing: list[str] = []
    used_authz = used_timing = 0
    for match in _SENTENCE_RE.finditer(tail):
        clause = match.group(0).strip()
        if not clause:
            continue
        # Authorization first: a sentence naming both a visa rule and a start
        # date is counted once, against the budget that matters more.
        if _is_authz_sentence(clause):
            if used_authz + len(clause) <= max_authz:
                authz.append(clause)
                used_authz += len(clause)
        elif _is_timing_sentence(clause):
            if used_timing + len(clause) <= max_timing:
                timing.append(clause)
                used_timing += len(clause)

    clauses = authz + timing
    if clauses:
        excerpt += " […] " + " ".join(clauses)
    return excerpt


def content_hash(title: str, location_raw: str | None) -> str:
    """Detects same-job-new-ID, which some ATS tenants produce on edit."""
    blob = f"{title.strip().lower()}|{(location_raw or '').strip().lower()}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def finalise(posting: Posting) -> Posting:
    posting.location = canonical_location(posting.location_raw)
    posting.remote = posting.remote or is_remote(posting.location_raw)
    posting.content_hash = content_hash(posting.title, posting.location_raw)
    return posting
