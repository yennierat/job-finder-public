"""Tests for confirming rejections before they become final.

The model is not deterministic at temperature 0: re-judging one posting against
an identical prompt was measured varying ~15 points, and one posting went
no / yes / yes across three consecutive calls. A single rejection is therefore
a coin toss, and caching it loses the job permanently.

The rule under test: a MATCH is settled immediately; a REJECTION must be
repeated before it sticks. These tests are mostly about the ways that rule
could quietly stop working — a count that resets, a match that gets
second-guessed, a retry loop with no bound.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import store  # noqa: E402
from src.models import CallMeta, Posting, Verdict  # noqa: E402

failures = []


def check(name, got, want):
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


def ok(name, cond):
    if not cond:
        failures.append(f"{name}: expected true")


PV, PH = "5", "hash"
META = CallMeta(model="m", attempt=0, fallback_depth=0, latency_ms=1,
                input_chars=1, output_chars=1)


def posting(external_id="1"):
    return Posting(source_id="src", external_id=external_id, title="Engineer",
                   location="Singapore", employment_type="internship")


def listed(conn, external_id="1"):
    """A posting that exists and is currently on a board."""
    p = posting(external_id)
    store.upsert_postings(conn, [p])
    return p


def judge(conn, p, is_match, score=50):
    store.record_classification(
        conn, p, Verdict(id=p.external_id, is_match=is_match, fit_score=score),
        META, PV, PH, run_id="r")


def pending(conn, ps):
    return [p.external_id for p in store.unclassified(conn, ps, PV, PH)]


tmp = tempfile.TemporaryDirectory()
conn = store.connect(Path(tmp.name) / "t.db")

# --- a rejection is not final until it is repeated ------------------------

p = listed(conn, "reject-once")
check("unjudged posting is pending", pending(conn, [p]), ["reject-once"])

judge(conn, p, False)
check("one rejection is not enough to settle it",
      pending(conn, [p]), ["reject-once"])
check("it is offered for a second opinion",
      store.awaiting_second_opinion(conn, PV, PH), [("src", "reject-once")])

judge(conn, p, False)
check("two rejections settle it", pending(conn, [p]), [])
check("and it stops being offered",
      store.awaiting_second_opinion(conn, PV, PH), [])

# The count must survive the write. INSERT OR REPLACE discards the old row, so
# a count read from the wrong place resets to 1 and the posting is re-judged
# every run forever — an unbounded LLM bill that looks like normal operation.
row = conn.execute(
    "SELECT times_classified FROM classifications WHERE external_id=?",
    ("reject-once",)).fetchone()
check("judgements are counted, not reset", row["times_classified"], 2)


# --- a match is settled immediately ---------------------------------------
# The errors are asymmetric: a false match costs one message you delete, a
# false rejection costs the job. So matches are never second-guessed.

m = listed(conn, "match-once")
judge(conn, m, True)
check("one match settles it", pending(conn, [m]), [])
check("a match is never offered for review",
      store.awaiting_second_opinion(conn, PV, PH), [])


# --- a second opinion can rescue a job ------------------------------------
# This is the whole point: the Optiver posting was rejected on one run and
# matched on the next, with nothing about it changed.

r = listed(conn, "rescued")
judge(conn, r, False, score=72)
check("rejected first", pending(conn, [r]), ["rescued"])
judge(conn, r, True, score=72)
check("matched on review, and now settled", pending(conn, [r]), [])
row = conn.execute("SELECT is_match FROM classifications WHERE external_id=?",
                   ("rescued",)).fetchone()
check("the match replaced the rejection", row["is_match"], 1)


# --- bounds ---------------------------------------------------------------

check("two judgements is the rule", store.CONFIRM_REJECTIONS, 2)

# Retries must be capped, or a large backlog spends the whole rate limit
# re-judging old rejections while genuinely new postings go unclassified.
for i in range(store.RETRY_LIMIT + 15):
    judge(conn, listed(conn, f"bulk-{i:03d}"), False)
offered = store.awaiting_second_opinion(conn, PV, PH)
check("retries are capped per run", len(offered), store.RETRY_LIMIT)

# Oldest first, so a backlog drains in order instead of the same rows coming up
# every run while the tail is never reached.
check("oldest first", offered[0], ("src", "bulk-000"))

# A different prompt or profile is a different question, so old verdicts must
# not answer it — and must not be dragged in as retries either.
check("another prompt version sees it as unjudged",
      [x.external_id for x in store.unclassified(conn, [p], "9", PH)],
      ["reject-once"])
check("another profile hash sees it as unjudged",
      [x.external_id for x in store.unclassified(conn, [p], PV, "other")],
      ["reject-once"])
check("retries are scoped to the current prompt",
      store.awaiting_second_opinion(conn, "9", PH), [])

# Rows written before this feature existed have no count. They must get one
# more look rather than being grandfathered in as confirmed.
legacy = listed(conn, "legacy")
judge(conn, legacy, False)
conn.execute("UPDATE classifications SET times_classified=NULL "
             "WHERE external_id=?", ("legacy",))
conn.commit()
check("a legacy row is re-judged, not trusted",
      pending(conn, [legacy]), ["legacy"])
ok("a legacy row is offered for review",
   ("src", "legacy") in store.awaiting_second_opinion(conn, PV, PH, limit=500))


# --- delisted postings must not block the queue ---------------------------
# A second opinion needs the advert text, and that only exists for postings
# still being fetched. A delisted one can never be re-judged, so offering it
# burns a slot on nothing — and because the queue drains oldest-first, dead
# rows would take every slot forever and no live posting would come up again.
# Measured before the fix: 40 of 40 slots dead, the live posting never offered.

# Its own database: the bulk rows above are live and unconfirmed, so they would
# fill the queue ahead of anything created here and mask what is being tested.
conn2 = store.connect(Path(tmp.name) / "t2.db")

gone = [posting(f"gone-{i:03d}") for i in range(store.RETRY_LIMIT + 5)]
store.upsert_postings(conn2, gone)
for g in gone:
    judge(conn2, g, False)
# These rows are the OLDEST rejections, so they sort to the front of the queue.
conn2.execute("UPDATE postings SET last_seen=datetime('now','-30 days') "
             "WHERE external_id LIKE 'gone-%'")
conn2.execute("UPDATE classifications SET created_at='2000-01-01T00:00:00+00:00' "
             "WHERE external_id LIKE 'gone-%'")
conn2.commit()

still_here = listed(conn2, "still-listed")
judge(conn2, still_here, False)

offered = store.awaiting_second_opinion(conn2, PV, PH)
check("delisted postings are not offered",
      [e for _, e in offered if e.startswith("gone-")], [])
ok("a still-listed posting gets its slot",
   ("src", "still-listed") in offered)

# The boundary: "still listed" means seen recently, not seen ever.
check("staleness is bounded in days", store.RETRY_MAX_STALE_DAYS, 2)
conn2.execute("UPDATE postings SET last_seen=datetime('now','-1 days') "
             "WHERE external_id=?", ("still-listed",))
conn2.commit()
ok("yesterday still counts as listed",
   ("src", "still-listed") in store.awaiting_second_opinion(conn2, PV, PH))
conn2.execute("UPDATE postings SET last_seen=datetime('now','-5 days') "
             "WHERE external_id=?", ("still-listed",))
conn2.commit()
ok("five days gone does not",
   ("src", "still-listed") not in store.awaiting_second_opinion(conn2, PV, PH))

# A rejection whose posting row has been pruned away entirely must not crash
# the query or resurface as a phantom retry.
conn2.execute("DELETE FROM postings WHERE external_id LIKE 'gone-%'")
conn2.commit()
check("orphaned rejections are simply absent",
      [e for _, e in store.awaiting_second_opinion(conn2, PV, PH, limit=500)
       if e.startswith("gone-")], [])

conn.close()
conn2.close()
tmp.cleanup()


if failures:
    print(f"{len(failures)} FAILURES:")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("reclassify tests passed")
