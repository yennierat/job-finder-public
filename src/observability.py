"""Structured JSON logging to stdout, plus run-summary helpers."""

import json
import sys
from datetime import UTC, datetime, timedelta


def log(event: str, **fields) -> None:
    record = {"ts": datetime.now(UTC).isoformat(), "event": event, **fields}
    print(json.dumps(record, default=str), flush=True)


def log_error(event: str, exc: BaseException, **fields) -> None:
    log(event, level="error", exc_type=type(exc).__name__, message=str(exc)[:500],
        **fields)


def summarise(conn, run_id: str) -> dict:
    row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return dict(row) if row else {}


def health_report(conn) -> str:
    """Daily heartbeat text. Absence of alerts must not be indistinguishable
    from absence of the system."""
    healthy = conn.execute(
        "SELECT COUNT(*) c FROM source_health WHERE consecutive_fails=0"
    ).fetchone()["c"]
    broken = conn.execute(
        "SELECT COUNT(*) c FROM source_health WHERE consecutive_fails>0"
    ).fetchone()["c"]
    today = conn.execute(
        "SELECT COUNT(*) c FROM notifications WHERE sent_at > datetime('now','-1 day')"
    ).fetchone()["c"]
    week = conn.execute(
        "SELECT COUNT(*) c FROM notifications WHERE sent_at > datetime('now','-7 days')"
    ).fetchone()["c"]
    postings = conn.execute("SELECT COUNT(*) c FROM postings").fetchone()["c"]
    # Still listed and still owed a verdict. Zero after a run that kept up; a
    # number that stays up is classification falling behind, which "0 matches
    # today" alone cannot tell apart from a quiet market. "Still listed" uses
    # the second-opinion window, since a delisted posting can never be judged.
    from src import store
    listed_since = (datetime.now(UTC)
                    - timedelta(days=store.RETRY_MAX_STALE_DAYS)).isoformat()
    awaiting = conn.execute(
        "SELECT COUNT(*) c FROM postings WHERE awaiting_verdict=1 AND last_seen >= ?",
        (listed_since,)).fetchone()["c"]
    return (f"still alive — {healthy} sources healthy, {broken} failing, "
            f"{postings} postings tracked, {awaiting} awaiting a verdict, "
            f"{today} matches today, {week} in the last 7 days")


HEARTBEAT_KEY = "last_heartbeat"
ALERT_KEY = "last_failure_alert"
CLASSIFY_ALERT_KEY = "last_classification_alert"
# Daily. The cron runs every 6h, so this fires on roughly the first run of each
# day rather than on all four.
HEARTBEAT_HOURS = 24
ALERT_COOLDOWN_HOURS = 24


# Below this many postings, a run returning nothing says more about the
# postings than about the models. One posting can poison its own batch and then
# fail again alone — classify.py retries item by item for exactly that reason —
# and a quiet 6-hourly run often has only one or two to judge. Three is a full
# batch: enough that every model in the chain has refused several times.
MIN_VERDICTS_TO_JUDGE_DOWN = 3


def classification_down(verdicts_requested: int, verdicts_returned: int) -> bool:
    """Postings were sent for a verdict and not one came back.

    Losing some is ordinary on free models, and they are retried next run.
    Losing all of them means classification itself is broken — a withdrawn free
    tier, an expired key — and nothing else notices. Sources stay healthy, the
    run finishes, and the heartbeat reports "0 matches today", which is exactly
    what a quiet week looks like.
    """
    return (verdicts_requested >= MIN_VERDICTS_TO_JUDGE_DOWN
            and verdicts_returned == 0)


def send_ops_messages(conn, notifier, sources_ok: int, sources_failed: int,
                      llm_failures: int = 0, verdicts_requested: int = 0,
                      verdicts_returned: int = 0) -> dict:
    """Heartbeat daily, and alert when a run is systemically broken.

    Without this the system fails silently: a disabled cron, an expired API key
    or every source breaking all look exactly like a quiet job market. The point
    is that absence of messages should mean 'nothing matched', never 'the
    monitor died three weeks ago'.
    """
    from src import store

    sent = {"heartbeat": False, "alert": False}
    total = sources_ok + sources_failed

    def cooled(key: str) -> bool:
        """Cooled down: a failure lasting a week must not produce 28 alerts."""
        since = store.hours_since(conn, key)
        return since is None or since >= ALERT_COOLDOWN_HOURS

    # A cooldown per condition rather than one shared between them. Boards
    # breaking and the model chain dying are unrelated failures; on one stamp,
    # whichever came first silenced the other for a day, and the message naming
    # the half that actually broke is the one that went missing.
    problems = []
    if total > 0 and (sources_failed / total) > 0.5:
        problems.append((ALERT_KEY,
                         f"{sources_failed} of {total} sources failed this run"))
    if classification_down(verdicts_requested, verdicts_returned):
        problems.append((CLASSIFY_ALERT_KEY,
                         f"no verdicts: all {verdicts_requested} postings sent for "
                         f"classification came back unjudged. They are retried "
                         f"next run, but nothing is being matched."))

    due = [(key, line) for key, line in problems if cooled(key)]
    if due:
        lines = ["job monitor DEGRADED"] + [line for _, line in due]
        if llm_failures:
            lines.append(f"{llm_failures} LLM call failures")
        if notifier.send_text("\n".join(lines)):
            for key, _ in due:
                store.stamp(conn, key)
            sent["alert"] = True

    since = store.hours_since(conn, HEARTBEAT_KEY)
    if since is None or since >= HEARTBEAT_HOURS:
        if notifier.send_text(f"job monitor heartbeat — {health_report(conn)}"):
            store.stamp(conn, HEARTBEAT_KEY)
            sent["heartbeat"] = True

    return sent


def exit_code(sources_ok: int, sources_failed: int, verdicts_requested: int = 0,
              verdicts_returned: int = 0) -> int:
    """Non-zero only on systemic failure, so a single broken board stays quiet.

    Systemic means most sources failing, or classification returning nothing at
    all. A run that judged none of what it fetched has not succeeded, and
    failing it is what fires the workflow's own alert, which unlike
    send_ops_messages has no cooldown to sit behind.
    """
    total = sources_ok + sources_failed
    if total == 0:
        return 1
    if classification_down(verdicts_requested, verdicts_returned):
        return 1
    return 1 if sources_failed / total > 0.5 else 0


def die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)
