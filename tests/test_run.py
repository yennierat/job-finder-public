"""Tests for the orchestration in run.py.

This module was untested, and both bugs it has shipped lived here — which is
not a coincidence. Neither was a crash: one was a dead comparison that could
not fire, the other a False return value falling through a loop. Both left the
run green and the log plausible, which is the only failure mode that matters
here.

No network: fetch and the notifier are replaced with fakes.
"""

import contextlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import run as run_mod  # noqa: E402
from src import store  # noqa: E402
from src.config import Source  # noqa: E402
from src.models import CallMeta, Posting, Verdict  # noqa: E402

failures = []


def check(name, got, want):
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


def ok(name, cond):
    if not cond:
        failures.append(f"{name}: expected true")


PV, PH = "9", "hash"
META = CallMeta(model="m", attempt=0, fallback_depth=0, latency_ms=1,
                input_chars=1, output_chars=1)


def posting(external_id="1", source_id="acme-workday"):
    return Posting(source_id=source_id, external_id=external_id,
                   title="Software Engineer Intern", location="Singapore",
                   employment_type="internship",
                   url=f"https://example.test/{external_id}")


def source(source_id="acme-workday"):
    return Source(id=source_id, name="acme", platform="workday",
                  board="acme/wd1/External")


@contextlib.contextmanager
def temp_db(name):
    """A throwaway database that is always closed.

    Closing it matters on Windows: TemporaryDirectory cannot remove an open
    SQLite file, so anything raising mid-block would bury the real failure
    under a PermissionError from the cleanup.
    """
    with tempfile.TemporaryDirectory() as tmp:
        conn = store.connect(Path(tmp) / name)
        try:
            yield conn
        finally:
            conn.close()


class CapturedLog:
    """Collect run.py's structured log instead of printing it."""

    def __init__(self):
        self.events = []

    def __enter__(self):
        self._log, self._log_error = run_mod.log, run_mod.log_error
        run_mod.log = lambda event, **f: self.events.append((event, f))
        run_mod.log_error = lambda event, exc, **f: self.events.append((event, f))
        return self

    def __exit__(self, *exc):
        run_mod.log, run_mod.log_error = self._log, self._log_error

    def fields(self, event):
        return [f for e, f in self.events if e == event]

    def saw(self, event):
        return any(e == event for e, _ in self.events)


# --- a board that goes empty must be noticed ------------------------------
# A tenant returning HTTP 200 with zero jobs never errors, so the circuit
# breaker cannot see it: consecutive_fails stays 0 and it is never
# quarantined. This log line is the only thing watching for it.
#
# It had never once fired. record_source_result overwrites last_count on the
# success path, and previous_count was read immediately AFTER it, so `before`
# came back equal to len(got) every time — and therefore 0 on exactly the runs
# the check existed to catch.

with temp_db("empty.db") as conn:

    def fetch_returning(items):
        return lambda platform, source_id, board: list(items)

    real_fetch = run_mod.fetch
    try:
        # Run 1: healthy, three postings.
        run_mod.fetch = fetch_returning([posting("a"), posting("b"),
                                         posting("c")])
        with CapturedLog() as first:
            got, okc, failed = run_mod.fetch_all([source()], conn, "run-1")
        check("healthy run returns its postings", len(got), 3)
        check("healthy run counts a success", okc, 1)
        ok("a healthy board is not reported empty", not first.saw("source.went_empty"))

        # Run 2: the tenant is dead. 200, no jobs, no exception.
        run_mod.fetch = fetch_returning([])
        with CapturedLog() as second:
            got, okc, failed = run_mod.fetch_all([source()], conn, "run-2")
        check("empty run still counts as a success", okc, 1)
        check("empty run raises nothing", failed, 0)
        fired = second.fields("source.went_empty")
        ok("going empty is reported", bool(fired))
        check("the previous count is named",
              fired[0].get("previous") if fired else None, 3)

        # And the new count is still recorded, so a third empty run is silent:
        # the board is now known to be empty rather than newly so.
        with CapturedLog() as third:
            run_mod.fetch_all([source()], conn, "run-3")
        ok("staying empty is not re-reported", not third.saw("source.went_empty"))

        # A board that was empty and comes back is not an event either.
        run_mod.fetch = fetch_returning([posting("a")])
        with CapturedLog() as fourth:
            run_mod.fetch_all([source()], conn, "run-4")
        ok("recovering is not reported as empty",
           not fourth.saw("source.went_empty"))
    finally:
        run_mod.fetch = real_fetch


# --- a posting that got no verdict must come back -------------------------
# upsert_postings stores a posting the moment it is first seen, so it is "new"
# exactly once. A run that then ends without classifying it — the budget spent,
# every model down, a crash, the job timeout — used to lose it for good: not
# new, not a rejection, and nothing else looking for it. Measured before the
# fix: run 2 offered nothing at all.
#
# PROMPT_VERSION comes from run.py rather than the PV above, because
# select_pending uses its own and the two must not drift apart in this file.

PVR = run_mod.PROMPT_VERSION

with temp_db("owed.db") as conn:
    p = posting("owed")

    # Run 1: new, queued for classification — and no verdict comes back.
    new = store.upsert_postings(conn, [p])
    check("run 1 sees it as new", [x.external_id for x in new], ["owed"])
    with CapturedLog():
        pending = run_mod.select_pending(conn, new, [p], PH)
    check("run 1 queues it", [x.external_id for x in pending], ["owed"])

    # Run 2: still on the board, but no longer new. This is the whole bug.
    new = store.upsert_postings(conn, [p])
    check("run 2 sees nothing new", new, [])
    with CapturedLog() as second:
        pending = run_mod.select_pending(conn, new, [p], PH)
    check("run 2 picks it up anyway", [x.external_id for x in pending], ["owed"])
    ok("and says so in the log", second.saw("classify.owed"))

    # A verdict discharges the debt, in the same write that records it.
    store.record_classification(conn, p, Verdict(id="owed", is_match=True),
                                META, PVR, PH, run_id="run-2")
    with CapturedLog():
        pending = run_mod.select_pending(conn, [], [p], PH)
    check("a judged posting is not queued again", pending, [])


with temp_db("owed-edges.db") as conn:
    # A posting that has left its board cannot be judged — run.py has no advert
    # text for it — so it is not offered. The mark stays for when it returns.
    gone = posting("delisted")
    store.upsert_postings(conn, [gone])
    with CapturedLog():
        run_mod.select_pending(conn, [gone], [gone], PH)
    with CapturedLog():
        check("a delisted posting is not offered",
              run_mod.select_pending(conn, [], [], PH), [])
    with CapturedLog():
        check("and is offered again when its board comes back",
              [x.external_id for x in run_mod.select_pending(conn, [], [gone], PH)],
              ["delisted"])

    # Seeding records what is already open and deliberately classifies none of
    # it. Those postings are owed nothing, or --seed would notify on the whole
    # backlog one run later — exactly what it exists to prevent.
    seeded = posting("seeded")
    store.upsert_postings(conn, [seeded])
    with CapturedLog():
        offered = run_mod.select_pending(conn, [], [gone, seeded], PH)
    check("a seeded posting is never owed a verdict",
          [x.external_id for x in offered], ["delisted"])

    # Two boards can number a job the same — Workday tenants both hand out
    # R-12345 — and classify() keys verdicts by external_id alone. Sent in one
    # run, the model answers that id once and both postings would take the
    # verdict, notifying for a job nobody judged. One goes, the other waits.
    dup_a = posting("R-12345", source_id="tenant-a")
    dup_b = posting("R-12345", source_id="tenant-b")
    store.upsert_postings(conn, [dup_a, dup_b])
    with CapturedLog() as collided:
        offered = run_mod.select_pending(conn, [dup_a, dup_b],
                                         [gone, dup_a, dup_b], PH)
    check("a shared id is only sent once",
          [x.external_id for x in offered].count("R-12345"), 1)
    ok("the collision is logged", collided.saw("classify.id_collision"))
    check("and the one held back is still owed a verdict",
          conn.execute("SELECT COUNT(*) c FROM postings WHERE external_id=? "
                       "AND awaiting_verdict=1", ("R-12345",)).fetchone()["c"], 2)

    # A rejection awaiting its second opinion is still picked up, and a posting
    # that is both owed and awaiting one is offered once, not twice.
    once = posting("rejected-once")
    store.upsert_postings(conn, [once])
    with CapturedLog():
        run_mod.select_pending(conn, [once], [once], PH)
    store.record_classification(conn, once, Verdict(id="rejected-once",
                                                    is_match=False),
                                META, PVR, PH, run_id="run-1")
    with CapturedLog():
        offered = run_mod.select_pending(conn, [], [gone, once], PH)
    check("a rejection still gets its second opinion",
          sorted(x.external_id for x in offered), ["delisted", "rejected-once"])
    check("and is offered exactly once", len(offered), 2)


with temp_db("owed-bounds.db") as conn:
    # The backlog is capped per run because everything pending is enriched — one
    # HTTP request per posting on the platforms that withhold advert text —
    # BEFORE the first model call, and enrichment has no budget of its own. An
    # outage grows this queue every run; uncapped, each later run would
    # re-enrich the whole of it before reaching a model.
    many = [posting(f"owed-{i:03d}") for i in range(store.OWED_LIMIT + 7)]
    store.upsert_postings(conn, many)
    store.mark_awaiting_verdict(conn, many)
    offered = store.awaiting_verdict(conn)
    check("the owed queue is capped per run", len(offered), store.OWED_LIMIT)
    check("oldest first", offered[0][1], "owed-000")

    # And a posting that has left its board keeps its mark but stops taking a
    # slot: it can never be judged while delisted, and the queue drains
    # oldest-first, so dead rows would hold every slot forever.
    conn.execute("UPDATE postings SET last_seen=datetime('now','-30 days') "
                 "WHERE external_id LIKE 'owed-0%'")
    conn.commit()
    fresh = store.awaiting_verdict(conn)
    check("delisted postings do not hold slots",
          [e for _, e in fresh if e.startswith("owed-0")], [])
    check("they keep the mark for when the board returns",
          conn.execute("SELECT COUNT(*) c FROM postings WHERE awaiting_verdict=1"
                       ).fetchone()["c"], store.OWED_LIMIT + 7)


# --- a refused send must not lose the job ---------------------------------
# notifier.send can return False without raising: Telegram does so after three
# transport failures, three 429s, or any non-400 status, with the plain-text
# fallback having failed too. That used to fall through the loop leaving no
# notification row, no log line and no error row — and because the verdict is
# settled by then, nothing would ever look at the posting again.

class Notifier:
    """Records what it was asked to send; delivers or refuses on command."""

    def __init__(self, result=True):
        self.result = result
        self.sent = []

    def send(self, posting, verdict):
        self.sent.append(posting.external_id)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def send_text(self, text):
        return True


with temp_db("deliver.db") as conn:

    p = posting("refused")
    store.upsert_postings(conn, [p])
    v = Verdict(id="refused", is_match=True, reason="Matches.", fit_score=80,
                matched_skills=["Python", "Docker"], missing_skills=["Rust"])
    store.record_classification(conn, p, v, META, PV, PH, run_id="run-1")

    refusing = Notifier(result=False)
    with CapturedLog() as log1:
        sent = run_mod.deliver(refusing, [(p, v)], conn, "run-1")

    check("a refusal delivers nothing", sent, 0)
    check("but the send was attempted", refusing.sent, ["refused"])
    check("no notification row is written",
          store.already_notified(conn, p.source_id, p.external_id), False)
    # The whole point: a refusal has to leave a trace. Silence here is what
    # made the original bug invisible.
    ok("the refusal is logged", log1.saw("notify.refused"))
    check("and recorded as an error",
          conn.execute("SELECT COUNT(*) c FROM errors WHERE stage='notify'"
                       ).fetchone()["c"], 1)

    # Because no notification row exists, the match is still owed a message.
    owed = store.undelivered_matches(conn, PV, PH)
    check("the match is queued for redelivery", [k for k, _ in owed],
          [("acme-workday", "refused")])
    # The verdict is rebuilt from storage, not re-derived from the model — a
    # redelivery must carry the same message the first attempt would have.
    _, rebuilt = owed[0]
    check("reason survives", rebuilt.reason, "Matches.")
    check("score survives", rebuilt.fit_score, 80)
    check("matched skills survive", rebuilt.matched_skills, ["Python", "Docker"])
    check("missing skills survive", rebuilt.missing_skills, ["Rust"])
    ok("rebuilt verdict is a match", rebuilt.is_match)

    # Next run, the notifier works.
    working = Notifier(result=True)
    with CapturedLog():
        sent = run_mod.deliver(working, [(p, rebuilt)], conn, "run-2")
    check("redelivery succeeds", sent, 1)
    check("and is recorded",
          store.already_notified(conn, p.source_id, p.external_id), True)
    check("so nothing remains owed", store.undelivered_matches(conn, PV, PH), [])

    # Exactly-once: a posting already notified is never sent twice, even if it
    # is handed to deliver() again.
    again = Notifier(result=True)
    with CapturedLog():
        check("a delivered match is not resent",
              run_mod.deliver(again, [(p, rebuilt)], conn, "run-3"), 0)
    check("the notifier was not called", again.sent, [])



# --- a raising notifier is handled separately -----------------------------

with temp_db("raise.db") as conn:
    p = posting("boom")
    store.upsert_postings(conn, [p])
    v = Verdict(id="boom", is_match=True)
    store.record_classification(conn, p, v, META, PV, PH, run_id="run-1")

    with CapturedLog() as log2:
        sent = run_mod.deliver(Notifier(result=RuntimeError("chat not found")),
                               [(p, v)], conn, "run-1")
    check("a raising notifier delivers nothing", sent, 0)
    ok("the exception is logged", log2.saw("notify.failed"))
    check("with a traceback recorded",
          conn.execute("SELECT exc_type FROM errors").fetchone()["exc_type"],
          "RuntimeError")
    # Same guarantee as a refusal: still owed, so still retried.
    check("a crash also leaves the match queued",
          [k for k, _ in store.undelivered_matches(conn, PV, PH)],
          [("acme-workday", "boom")])

    # One bad posting must not stop the rest of the batch.
    good = posting("fine")
    store.upsert_postings(conn, [good])
    store.record_classification(conn, good, Verdict(id="fine", is_match=True),
                                META, PV, PH, run_id="run-2")
    partial = Notifier(result=True)
    with CapturedLog():
        sent = run_mod.deliver(partial, [(p, v), (good, Verdict(id="fine",
                                                                is_match=True))],
                               conn, "run-2")
    check("the rest of the batch still goes out", sent, 2)


# --- rejections are never owed a message ----------------------------------

with temp_db("reject.db") as conn:
    p = posting("rejected")
    store.upsert_postings(conn, [p])
    store.record_classification(conn, p, Verdict(id="rejected", is_match=False),
                                META, PV, PH, run_id="run-1")
    check("a rejection is not queued for delivery",
          store.undelivered_matches(conn, PV, PH), [])

    # Nor is a match judged under a different prompt or profile: that verdict
    # answers a question no longer being asked.
    m = posting("other-prompt")
    store.upsert_postings(conn, [m])
    store.record_classification(conn, m, Verdict(id="other-prompt", is_match=True),
                                META, "1", PH, run_id="run-1")
    check("another prompt version is not queued",
          store.undelivered_matches(conn, PV, PH), [])
    check("another profile hash is not queued",
          store.undelivered_matches(conn, "1", "elsewhere"), [])
    check("its own prompt version still finds it",
          [k for k, _ in store.undelivered_matches(conn, "1", PH)],
          [("acme-workday", "other-prompt")])

    # A posting pruned out from under its verdict must not resurface.
    conn.execute("DELETE FROM postings WHERE external_id=?", ("other-prompt",))
    conn.commit()
    check("an orphaned verdict is not queued",
          store.undelivered_matches(conn, "1", PH), [])

    # The queue is bounded, so a week of failed delivery cannot produce a burst
    # that earns a 429 from Telegram.
    for i in range(store.REDELIVER_LIMIT + 8):
        b = posting(f"bulk-{i:03d}")
        store.upsert_postings(conn, [b])
        store.record_classification(conn, b, Verdict(id=b.external_id,
                                                     is_match=True),
                                    META, PV, PH, run_id="run-2")
    check("redelivery is capped per run",
          len(store.undelivered_matches(conn, PV, PH)), store.REDELIVER_LIMIT)
    check("oldest first",
          store.undelivered_matches(conn, PV, PH)[0][0][1], "bulk-000")


if failures:
    print(f"{len(failures)} FAILURES:")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("run tests passed")
