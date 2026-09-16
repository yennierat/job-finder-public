"""Offline tests: normalisation, prefilter, dedupe, and verdict matching.

No network and no LLM — these run anywhere, fast.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import store  # noqa: E402
from src.models import CallMeta, JobProfile, Posting, Verdict  # noqa: E402
from src.normalise import (_is_authz_sentence, canonical_location,  # noqa: E402
                           content_hash, description_excerpt,
                           employment_type, finalise, html_to_text)
from src.prefilter import Prefilter  # noqa: E402

failures = []


def check(name, got, want):
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


# --- location canonicalisation -------------------------------------------
for raw, want in [
    ("Singapore", "Singapore"),
    ("SG - Singapore", "Singapore"),
    ("Singapore, Singapore", "Singapore"),
    ("Asia Pacific - Singapore", "Singapore"),
    ("Hong Kong, Hong Kong", "Hong Kong"),
    ("HK", "Hong Kong"),
    ("San Francisco", "San Francisco"),
    ("Chicago, Illinois", "Chicago"),
    ("Genovia", None),          # unknown stays unknown, never "no match"
    (None, None),
]:
    check(f"canonical_location({raw!r})", canonical_location(raw), want)

# --- employment type ------------------------------------------------------
for title, structured, want in [
    ("Software Engineer Intern", None, "internship"),
    ("Campus Python Software Engineer (Intern)", None, "internship"),
    ("Summer Analyst Internship 2027", None, "internship"),
    ("Graduate Programme - Technology", None, "graduate_programme"),
    ("New Grad Software Engineer", None, "graduate_programme"),
    ("Senior Backend Engineer", None, "full_time"),
    ("Data Analyst", "Contract", "contract"),
    ("Support Agent", "Part-Time", "contract"),
    ("Trading Intern", "Full-Time", "internship"),  # title beats a wrong field
    # Banks say "Summer Analyst", never "intern".
    ("Banking - Corporate Banking, Summer Analyst, Singapore", None, "internship"),
    ("Markets - Sales and Trading, Summer Analyst, Hong Kong", None, "internship"),
    ("Summer Associate, Investment Banking", None, "internship"),
    ("Off-Cycle Analyst, Equities", None, "internship"),
    ("Industrial Placement, Technology", None, "internship"),
    # One-week tasters are NOT internships.
    ("Spring Week Insight Programme", None, "full_time"),
    ("Vacation Scheme 2027", None, "full_time"),
    # A real analyst job must not be swept in.
    ("Quantitative Risk Analyst", None, "full_time"),
    ("Senior Data Analyst", None, "full_time"),
]:
    check(f"employment_type({title!r})", employment_type(title, structured), want)

# --- html to text ---------------------------------------------------------
check("strips tags", html_to_text("<p>Hello <b>world</b></p>"), "Hello world")
# Greenhouse double-escapes: "&lt;p&gt;" must not survive as literal text.
check("handles double-escaped html",
      html_to_text("&lt;p&gt;Intern &amp; grad&lt;/p&gt;"), "Intern & grad")
check("empty html is None", html_to_text(""), None)
check("None html is None", html_to_text(None), None)

# --- work-authorization sentence detection -------------------------------
for sentence, want in [
    # "Visa" the payment network / investor must NOT read as immigration text.
    ("Backed by Visa, Mastercard, Robinhood Ventures and Sequoia", False),
    ("Our platform processes Visa and Mastercard transactions daily", False),
    ("We welcome applicants regardless of race, religion or national origin", False),
    # EEO boilerplate lists citizenship among protected traits — not a visa rule.
    ("We don't regard color, religion, race, national origin, sexual orientation, "
     "ancestry, citizenship, sex, marital or family status", False),
    ("Equal opportunity employer regardless of citizenship, age or disability", False),
    ("You will need to obtain the required visa and/or permit to work here", True),
    ("We do not provide visa sponsorship for this role", True),
    ("Applicants must have the right to work in Singapore", True),
    ("Candidates must be eligible to work in the UK without sponsorship", True),
    ("This role requires a valid Employment Pass", True),
    ("Open to Singapore citizens and permanent residents only", True),
    ("You must be legally authorized to work in the United States", True),
]:
    check(f"authz({sentence[:38]!r})", _is_authz_sentence(sentence), want)

# The head is kept, and a visa clause far past the cutoff is appended rather
# than truncated away — the whole reason the excerpt is not a plain head slice.
long_desc = ("About the role. " * 60) + "You must be eligible to work in Singapore."
excerpt = description_excerpt(long_desc, head=200)
check("excerpt keeps head", excerpt.startswith("About the role."), True)
check("excerpt appends authz clause", "eligible to work in Singapore" in excerpt, True)
check("excerpt marks the elision", "[…]" in excerpt, True)
check("no description stays None", description_excerpt(None), None)
# Nothing to append means no elision marker.
check("plain description has no marker",
      "[…]" in description_excerpt("Short role description.", head=200), False)


# --- content hash detects same-job-new-id --------------------------------
check("content_hash stable",
      content_hash("SWE Intern", "Singapore")
      == content_hash("swe intern ", " singapore"),
      True)
check("content_hash differs on location",
      content_hash("SWE Intern", "Singapore")
      != content_hash("SWE Intern", "Hong Kong"),
      True)


# --- prefilter ------------------------------------------------------------
def posting(title, loc, etype="internship", remote=False):
    p = Posting(source_id="s", external_id=title, title=title,
                location_raw=loc, employment_type=etype, remote=remote)
    return finalise(p)


profile = JobProfile(
    employment_types=["internship"],
    locations=["Singapore", "Hong Kong"],
    must_have_keywords=["engineer", "developer"],
    exclude_keywords=["sales"],
)
pf = Prefilter(profile)

check("keeps SG intern",
      pf.keep(posting("Software Engineer Intern", "Singapore")), True)
check("keeps HK intern",
      pf.keep(posting("Backend Developer Intern", "Hong Kong")), True)
check("drops wrong city", pf.keep(posting("Software Engineer Intern", "London")), False)
check("drops full-time",
      pf.keep(posting("Software Engineer", "Singapore", "full_time")), False)
check("drops excluded keyword",
      pf.keep(posting("Sales Engineer Intern", "Singapore")), False)
check("drops missing must-have",
      pf.keep(posting("Marketing Intern", "Singapore")), False)
# Unknown location must survive: unknown != elsewhere.
check("keeps unknown location",
      pf.keep(posting("Software Engineer Intern", "Planet Zorg")), True)

# must_have is an OR — one hit is enough, not all.
check("must_have is OR", pf.keep(posting("Developer Intern", "Singapore")), True)

# remote_ok opens the location gate
remote_profile = JobProfile(employment_types=["internship"],
                            locations=["Singapore"], remote_ok=True)
check("remote allowed when remote_ok",
      Prefilter(remote_profile).keep(posting("Engineer Intern", "Remote", remote=True)),
      True)

# empty profile keeps everything
check("empty profile keeps all",
      Prefilter(JobProfile(employment_types=[])).keep(posting("Anything", "Nowhere")),
      True)

# regions expand to cities — "Europe" must match a London posting
region_profile = JobProfile(employment_types=["internship"],
                            locations=["Europe", "Singapore"])
rp = Prefilter(region_profile)
check("Europe matches London", rp.keep(posting("SWE Intern", "London, UK")), True)
check("Europe matches Amsterdam",
      rp.keep(posting("SWE Intern", "Amsterdam, Netherlands")), True)
check("Europe still excludes Tokyo", rp.keep(posting("SWE Intern", "Tokyo")), False)
check("named city alongside region still works",
      rp.keep(posting("SWE Intern", "SG - Singapore")), True)

apac = Prefilter(JobProfile(employment_types=["internship"], locations=["APAC"]))
check("APAC matches Hong Kong", apac.keep(posting("SWE Intern", "Hong Kong")), True)
check("APAC excludes London", apac.keep(posting("SWE Intern", "London")), False)


# --- store: dedupe + notification idempotency ----------------------------
with tempfile.TemporaryDirectory() as tmp:
    conn = store.connect(Path(tmp) / "t.db")

    a = posting("SWE Intern", "Singapore")
    b = posting("Data Intern", "Hong Kong")

    check("first upsert is all new", len(store.upsert_postings(conn, [a, b])), 2)
    check("second upsert is none new", len(store.upsert_postings(conn, [a, b])), 0)

    # same id, changed title -> treated as new (same-job-new-id detection)
    moved = Posting(source_id="s", external_id="SWE Intern",
                    title="SWE Intern", location_raw="Hong Kong")
    finalise(moved)
    check("content change re-notifies", len(store.upsert_postings(conn, [moved])), 1)

    check("unclassified initially", len(store.unclassified(conn, [a, b], "1", "h")), 2)

    # The real CallMeta, not a stand-in: a stub with the same attribute names
    # would keep passing after a field was renamed out from under
    # record_classification, which is exactly the break worth catching.
    meta = CallMeta(model="m", provider="p", latency_ms=1, attempt=0,
                    fallback_depth=0, input_chars=1, output_chars=1,
                    finish_reason="stop", session_id="sess-1",
                    generation_id="gen-1")

    store.record_classification(conn, a, Verdict(id=a.external_id, is_match=True),
                                meta, "1", "h", run_id="run-1")
    check("classified excluded", len(store.unclassified(conn, [a, b], "1", "h")), 1)
    # a different profile_hash must invalidate the cached verdict
    check("profile change invalidates",
          len(store.unclassified(conn, [a, b], "1", "OTHER")), 2)

    check("not notified yet", store.already_notified(conn, "s", a.external_id), False)
    store.mark_notified(conn, "s", a.external_id, run_id="run-1")
    check("notified now", store.already_notified(conn, "s", a.external_id), True)

    # tracing columns are actually persisted, not silently dropped
    row = conn.execute("SELECT run_id, session_id, generation_id, "
                       "work_authorization, authorization_quote FROM "
                       "classifications").fetchone()
    check("classification run_id stored", row["run_id"], "run-1")
    check("classification session_id stored", row["session_id"], "sess-1")
    check("classification generation_id stored", row["generation_id"], "gen-1")
    check("work_authorization defaults", row["work_authorization"], "not_mentioned")
    check("authorization_quote defaults empty", row["authorization_quote"], "")

    nrow = conn.execute("SELECT run_id FROM notifications").fetchone()
    check("notification run_id stored", nrow["run_id"], "run-1")

    # circuit breaker
    for _ in range(5):
        store.record_source_result(conn, "flaky", False)
    check("quarantined after 5 fails", store.is_quarantined(conn, "flaky"), True)
    store.record_source_result(conn, "flaky", True, 10)
    check("recovers on success", store.is_quarantined(conn, "flaky"), False)
    conn.close()


# --- prune: postings go, notifications never do --------------------------
with tempfile.TemporaryDirectory() as tmp:
    conn = store.connect(Path(tmp) / "p.db")
    old = posting("Old Intern", "Singapore")
    new = posting("New Intern", "Singapore")
    store.upsert_postings(conn, [old, new])

    meta = CallMeta(model="m", provider="p", latency_ms=1, attempt=0,
                    fallback_depth=0, input_chars=1, output_chars=1,
                    finish_reason="stop")

    store.record_classification(conn, old,
                                Verdict(id=old.external_id, is_match=True),
                                meta, "1", "h")
    store.mark_notified(conn, old.source_id, old.external_id)

    # Backdate one posting past the retention window.
    conn.execute("UPDATE postings SET last_seen = datetime('now','-200 days') "
                 "WHERE external_id=?", (old.external_id,))
    conn.commit()

    result = store.prune(conn, posting_days=90)
    check("prune reports count", result["postings_pruned"], 1)
    check("stale posting deleted",
          conn.execute("SELECT COUNT(*) c FROM postings").fetchone()["c"], 1)
    check("its classification deleted",
          conn.execute("SELECT COUNT(*) c FROM classifications").fetchone()["c"], 0)
    # The whole point: the notification outlives the posting, so a repost of the
    # same id is still suppressed.
    check("notification survives prune",
          conn.execute("SELECT COUNT(*) c FROM notifications").fetchone()["c"], 1)
    check("still suppresses a repost",
          store.already_notified(conn, old.source_id, old.external_id), True)
    conn.close()


# --- profile_hash ---------------------------------------------------------
from src.config import profile_hash  # noqa: E402

p1 = JobProfile(employment_types=["internship"], locations=["Singapore"])
p2 = JobProfile(employment_types=["internship"], locations=["Singapore"],
                inferred_fields=["locations"])
check("inferred_fields does not change hash", profile_hash(p1), profile_hash(p2))
p3 = JobProfile(employment_types=["internship"], locations=["Hong Kong"])
check("locations change hash", profile_hash(p1) != profile_hash(p3), True)


if failures:
    print(f"{len(failures)} FAILURES:")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("all tests passed")
