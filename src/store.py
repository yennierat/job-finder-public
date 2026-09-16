"""All SQL lives here. SQLite state: postings, verdicts, notifications, tracing."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.config import DB_PATH
from src.models import Posting, Verdict

SCHEMA = """
CREATE TABLE IF NOT EXISTS postings (
    source_id TEXT, external_id TEXT, title TEXT, url TEXT, location TEXT,
    content_hash TEXT, first_seen TEXT, last_seen TEXT,
    employment_type TEXT, location_raw TEXT, remote INTEGER,
    -- Application closing date, ISO. Persisted (unlike description, which is
    -- large and single-use) because it is a few bytes and it is what decides
    -- whether a match still matters tomorrow.
    deadline TEXT, deadline_text TEXT,
    -- 1 from the moment a posting is queued for classification until a verdict
    -- is recorded for it. A posting stops being "new" as soon as it is stored,
    -- so without this, one that a run failed to classify was never looked at
    -- again. Seeded postings never set it: they are owed nothing.
    awaiting_verdict INTEGER DEFAULT 0,
    PRIMARY KEY (source_id, external_id)
);
CREATE TABLE IF NOT EXISTS classifications (
    source_id TEXT, external_id TEXT, prompt_version TEXT, profile_hash TEXT,
    is_match INTEGER, category TEXT, reason TEXT, created_at TEXT,
    model TEXT, provider TEXT, latency_ms INTEGER, attempt INTEGER,
    fallback_depth INTEGER, input_chars INTEGER, output_chars INTEGER,
    finish_reason TEXT, run_id TEXT, session_id TEXT, generation_id TEXT,
    work_authorization TEXT, authorization_quote TEXT,
    -- NULL when no resume was configured for the run. Not 0: "not scored" and
    -- "scored zero" are different facts and must stay distinguishable.
    fit_score INTEGER, matched_skills TEXT, missing_skills TEXT,
    -- How many times this posting has been judged under this exact prompt and
    -- profile. A rejection is not final until CONFIRM_REJECTIONS of them agree.
    times_classified INTEGER DEFAULT 1,
    PRIMARY KEY (source_id, external_id, prompt_version, profile_hash)
);
-- Never pruned. It is the exactly-once guarantee, it is tiny, and a row
-- surviving its posting is the point: if a posting is pruned and later
-- reappears, this is what stops a duplicate notification.
CREATE TABLE IF NOT EXISTS notifications (
    source_id TEXT, external_id TEXT, sent_at TEXT, run_id TEXT,
    PRIMARY KEY (source_id, external_id)
);
CREATE TABLE IF NOT EXISTS llm_calls (
    run_id TEXT, stage TEXT, model TEXT, provider TEXT, attempt INTEGER,
    status TEXT, http_status INTEGER, latency_ms INTEGER, batch_size INTEGER,
    parsed_ok INTEGER, raw_response_excerpt TEXT, created_at TEXT,
    session_id TEXT, generation_id TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT,
    profile_hash TEXT, prompt_version TEXT, sources_ok INTEGER,
    sources_failed INTEGER, postings_seen INTEGER, postings_new INTEGER,
    llm_calls INTEGER, llm_failures INTEGER, notifications_sent INTEGER,
    mode TEXT, session_id TEXT
);
CREATE TABLE IF NOT EXISTS errors (
    run_id TEXT, source_id TEXT, stage TEXT, exc_type TEXT,
    message TEXT, traceback TEXT, occurred_at TEXT
);
-- Small key/value store, used to rate-limit ops messages so a persistent
-- failure alerts once a day rather than every six hours.
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS source_health (
    source_id TEXT PRIMARY KEY, consecutive_fails INTEGER DEFAULT 0,
    quarantined_until TEXT, last_success TEXT, last_count INTEGER
);
"""


def now() -> str:
    return datetime.now(UTC).isoformat()


def connect(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn) -> None:
    """Add columns missing from databases created by an earlier schema."""
    have = {r["name"] for r in conn.execute("PRAGMA table_info(postings)")}
    # awaiting_verdict defaults to 0 on existing rows: nothing on file was ever
    # marked as owed, so nothing is — rather than every unclassified posting in
    # the database, seeded ones included, suddenly queuing for the model.
    for column, ddl in (("employment_type", "TEXT"), ("location_raw", "TEXT"),
                        ("remote", "INTEGER"), ("deadline", "TEXT"),
                        ("deadline_text", "TEXT"),
                        ("awaiting_verdict", "INTEGER DEFAULT 0")):
        if column not in have:
            conn.execute(f"ALTER TABLE postings ADD COLUMN {column} {ddl}")

    have = {r["name"] for r in conn.execute("PRAGMA table_info(llm_calls)")}
    for column in ("session_id", "generation_id"):
        if column not in have:
            conn.execute(f"ALTER TABLE llm_calls ADD COLUMN {column} TEXT")

    have = {r["name"] for r in conn.execute("PRAGMA table_info(classifications)")}
    for column in ("run_id", "session_id", "generation_id", "work_authorization",
                   "authorization_quote", "matched_skills", "missing_skills"):
        if column not in have:
            conn.execute(f"ALTER TABLE classifications ADD COLUMN {column} TEXT")
    if "fit_score" not in have:
        conn.execute("ALTER TABLE classifications ADD COLUMN fit_score INTEGER")
    if "times_classified" not in have:
        # Existing rows default to 1, so every rejection already on file gets
        # exactly one more look rather than being grandfathered as confirmed.
        conn.execute("ALTER TABLE classifications "
                     "ADD COLUMN times_classified INTEGER DEFAULT 1")

    have = {r["name"] for r in conn.execute("PRAGMA table_info(notifications)")}
    if "run_id" not in have:
        conn.execute("ALTER TABLE notifications ADD COLUMN run_id TEXT")

    have = {r["name"] for r in conn.execute("PRAGMA table_info(runs)")}
    for column in ("mode", "session_id"):
        if column not in have:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {column} TEXT")
    conn.commit()


# --- postings -------------------------------------------------------------

def upsert_postings(conn, postings: list[Posting]) -> list[Posting]:
    """Insert/refresh postings; return those never seen before.

    A changed content_hash on a known id counts as new: some ATS tenants reuse
    or regenerate ids, so title+location shifting under a stable id is a
    different job, not an update.
    """
    fresh = []
    ts = now()
    for p in postings:
        row = conn.execute(
            "SELECT content_hash FROM postings WHERE source_id=? AND external_id=?",
            (p.source_id, p.external_id),
        ).fetchone()
        if row is None:
            # Named columns, not positional. A migrated database appends new
            # columns at the end while a fresh one gets them in schema order, so
            # a positional INSERT writes different fields depending on how the
            # file came to exist — silently, and only for some users.
            conn.execute(
                "INSERT INTO postings (source_id, external_id, title, url,"
                " location, content_hash, first_seen, last_seen,"
                " employment_type, location_raw, remote)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (p.source_id, p.external_id, p.title, p.url, p.location,
                 p.content_hash, ts, ts, p.employment_type, p.location_raw,
                 int(bool(p.remote))),
            )
            fresh.append(p)
        else:
            conn.execute(
                "UPDATE postings SET title=?, url=?, location=?, content_hash=?, "
                "last_seen=?, employment_type=?, location_raw=?, remote=? "
                "WHERE source_id=? AND external_id=?",
                (p.title, p.url, p.location, p.content_hash, ts,
                 p.employment_type, p.location_raw, int(bool(p.remote)),
                 p.source_id, p.external_id),
            )
            if row["content_hash"] != p.content_hash:
                fresh.append(p)
    conn.commit()
    return fresh


def save_deadlines(conn, postings: list[Posting]) -> int:
    """Persist deadlines found during enrichment.

    Separate from upsert_postings because enrichment runs after it: the
    prefilter decides which postings are worth a detail request, and it cannot
    run until the postings exist.
    """
    n = 0
    for p in postings:
        if not p.deadline:
            continue
        conn.execute(
            "UPDATE postings SET deadline=?, deadline_text=? "
            "WHERE source_id=? AND external_id=?",
            (p.deadline, p.deadline_text, p.source_id, p.external_id))
        n += 1
    conn.commit()
    return n


# --- classifications ------------------------------------------------------

# How many independent rejections it takes before a posting is dropped for good.
#
# The model is not deterministic even at temperature 0: re-scoring one posting
# against an identical prompt was measured varying by ~15 points, and verdicts
# flip near the boundary — one job went no / yes / yes across three consecutive
# calls. With a single judgement, a job you wanted is lost to a coin toss and
# never reconsidered.
#
# Two rather than three, because the errors are asymmetric: a false match costs
# one Telegram message you delete, a false rejection costs the job. So a match
# is acted on immediately and never second-guessed, while a rejection has to be
# repeated before it is believed.
CONFIRM_REJECTIONS = 2

# Cap on second opinions per run, so a backlog cannot spend the whole rate limit
# re-judging old rejections instead of classifying genuinely new postings.
#
# Twelve, not forty. Introducing this feature made every rejection already on
# file eligible at once — 341 of them — and at forty per run that tripled the
# classification load in the same week the primary model's free tier was
# withdrawn, turning a slow run into one killed by the job timeout. Twelve
# drains the backlog over a few weeks while leaving most of each run's budget
# for postings nobody has judged at all.
RETRY_LIMIT = 12

# A second opinion needs the advert text, which only exists for postings still
# being fetched. A delisted posting can therefore never be re-judged — and
# because the queue is ordered oldest-first, dead rows would otherwise occupy
# every slot forever and no live posting would ever come up. Measured: 40 of 40.
RETRY_MAX_STALE_DAYS = 2


def unclassified(conn, postings: list[Posting], prompt_version: str,
                 profile_hash: str) -> list[Posting]:
    """Postings still needing a verdict for this (prompt_version, profile_hash).

    That means either no verdict at all, or a rejection that has not yet been
    repeated CONFIRM_REJECTIONS times. A match is settled on the first verdict.

    Classification failures leave no row. That alone does not bring a posting
    back — only new postings are ever passed in here — which is what
    awaiting_verdict is for.
    """
    out = []
    for p in postings:
        row = conn.execute(
            "SELECT is_match, times_classified FROM classifications "
            "WHERE source_id=? AND external_id=? "
            "AND prompt_version=? AND profile_hash=?",
            (p.source_id, p.external_id, prompt_version, profile_hash),
        ).fetchone()
        if row is None or _needs_second_opinion(row):
            out.append(p)
    return out


def mark_awaiting_verdict(conn, postings: list[Posting]) -> None:
    """Record that these postings are owed a verdict, before any is attempted.

    upsert_postings has already stored them, so they will never be "new" again.
    If the run then ends without classifying one — the budget runs out, every
    model is down, a crash, the job timeout — this mark is the only thing that
    brings it back. record_classification clears it.
    """
    conn.executemany(
        "UPDATE postings SET awaiting_verdict=1 WHERE source_id=? AND external_id=?",
        [(p.source_id, p.external_id) for p in postings])
    conn.commit()


# Cap on the owed backlog offered per run. Higher than RETRY_LIMIT, because
# these postings have never been judged at all and a rejection awaiting
# confirmation already has a usable verdict on file.
#
# Capped rather than unbounded, and the reason is enrichment rather than
# classification. Everything pending is enriched BEFORE the first model call —
# one HTTP request per posting on the platforms that withhold advert text — and
# enrich() has no budget of its own, while classify()'s only starts afterwards.
# An outage marks every run's prefilter survivors as owed and judges none, so
# the backlog grows monotonically; uncapped, each later run would re-enrich all
# of it before reaching a model. That is the shape of the incident recorded
# against RETRY_LIMIT above.
OWED_LIMIT = 100


def awaiting_verdict(conn, limit: int = OWED_LIMIT) -> list[tuple[str, str]]:
    """(source_id, external_id) of postings owed a verdict by an earlier run.

    Keys rather than Postings, for the same reason as awaiting_second_opinion:
    run.py resolves them against what it fetched this run, so the posting is
    judged on its advert text, and one that has left its board is skipped.

    Oldest first, so a backlog drains in order rather than the same rows coming
    up every run. A delisted posting keeps its mark but is never resolved, so
    it would hold a slot forever — hence the same staleness bound the
    second-opinion queue uses.
    """
    cutoff = (datetime.now(UTC)
              - timedelta(days=RETRY_MAX_STALE_DAYS)).isoformat()
    rows = conn.execute(
        "SELECT source_id, external_id FROM postings WHERE awaiting_verdict=1 "
        "AND last_seen >= ? ORDER BY first_seen LIMIT ?", (cutoff, limit)).fetchall()
    return [(r["source_id"], r["external_id"]) for r in rows]


def _needs_second_opinion(row) -> bool:
    return (not row["is_match"]
            and (row["times_classified"] or 1) < CONFIRM_REJECTIONS)


def awaiting_second_opinion(conn, prompt_version: str, profile_hash: str,
                            limit: int = RETRY_LIMIT) -> list[tuple[str, str]]:
    """(source_id, external_id) of rejections that are not yet confirmed.

    Returns keys rather than Postings on purpose. run.py resolves them against
    the postings it has just fetched, so a re-judgement is made on the same
    advert text as the first one — a second opinion formed on less information
    than the first would be worse than none.

    Only postings still listed on a board are offered. One that has been
    delisted cannot be re-judged at all — run.py has no advert text for it — so
    including it would burn a slot on nothing. Since the queue drains
    oldest-first, dead rows would take every slot permanently and no live
    posting would ever be reconsidered.

    Oldest first, so a backlog drains in order instead of the same rows being
    retried every run while others never come up.
    """
    cutoff = (datetime.now(UTC)
              - timedelta(days=RETRY_MAX_STALE_DAYS)).isoformat()
    rows = conn.execute(
        "SELECT c.source_id, c.external_id FROM classifications c "
        "JOIN postings p ON p.source_id = c.source_id "
        " AND p.external_id = c.external_id "
        "WHERE c.prompt_version=? AND c.profile_hash=? AND c.is_match=0 "
        "AND COALESCE(c.times_classified, 1) < ? AND p.last_seen >= ? "
        "ORDER BY c.created_at LIMIT ?",
        (prompt_version, profile_hash, CONFIRM_REJECTIONS, cutoff, limit),
    ).fetchall()
    return [(r["source_id"], r["external_id"]) for r in rows]


def record_classification(conn, posting: Posting, verdict, meta,
                          prompt_version: str, profile_hash: str,
                          run_id: str = "") -> None:
    # INSERT OR REPLACE discards the old row, so the count has to be carried
    # forward explicitly or every re-judgement would reset it to 1 and the
    # posting would be re-classified forever.
    previous = conn.execute(
        "SELECT times_classified FROM classifications WHERE source_id=? AND "
        "external_id=? AND prompt_version=? AND profile_hash=?",
        (posting.source_id, posting.external_id, prompt_version, profile_hash),
    ).fetchone()
    times = (previous["times_classified"] or 0) + 1 if previous else 1

    # Named columns, not positional: this table has grown twice already, and a
    # positional INSERT silently shifts every value when a column is added.
    conn.execute(
        "INSERT OR REPLACE INTO classifications ("
        "source_id, external_id, prompt_version, profile_hash, is_match,"
        "category, reason, created_at, model, provider, latency_ms, attempt,"
        "fallback_depth, input_chars, output_chars, finish_reason, run_id,"
        "session_id, generation_id, work_authorization, authorization_quote,"
        "fit_score, matched_skills, missing_skills, times_classified"
        ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (posting.source_id, posting.external_id, prompt_version, profile_hash,
         int(verdict.is_match), verdict.category, verdict.reason, now(),
         meta.model, meta.provider, meta.latency_ms, meta.attempt,
         meta.fallback_depth, meta.input_chars, meta.output_chars,
         meta.finish_reason, run_id, meta.session_id, meta.generation_id,
         verdict.work_authorization, verdict.authorization_quote,
         verdict.fit_score,
         ", ".join(verdict.matched_skills), ", ".join(verdict.missing_skills),
         times),
    )
    # Same commit as the verdict, so a posting can never be both judged and
    # still owed, nor lose its mark without being judged.
    conn.execute(
        "UPDATE postings SET awaiting_verdict=0 WHERE source_id=? AND external_id=?",
        (posting.source_id, posting.external_id))
    conn.commit()


# --- notifications --------------------------------------------------------

# Cap on redeliveries per run. Higher than RETRY_LIMIT because these cost no
# model call at all, only a Telegram message, and Telegram tolerates ~20 per
# minute to one chat. A backlog this large means delivery has been broken for
# days, and draining it over a few runs is better than a burst that earns a 429.
REDELIVER_LIMIT = 20


def undelivered_matches(conn, prompt_version: str, profile_hash: str,
                        limit: int = REDELIVER_LIMIT
                        ) -> list[tuple[tuple[str, str], Verdict]]:
    """Matches that were judged but never delivered, oldest first.

    A send can fail WITHOUT raising: TelegramNotifier.send returns False after
    three transport failures, three 429s, or any non-400 status, and only after
    the plain-text fallback has failed too. By that point the posting is settled
    — it is no longer new, and `unclassified` excludes a confirmed match — so
    nothing downstream would ever look at it again and the job is lost silently.

    The notifications table is the exactly-once guarantee, so its absence is
    exactly the right signal: a row appears only after a confirmed send, which
    makes "has a match verdict and no notification" precisely the set still owed
    a message. Rows are returned as keys plus a rebuilt Verdict, and run.py
    resolves the keys against what it fetched this run, so a message is only
    ever sent for a posting that is still live.
    """
    rows = conn.execute(
        "SELECT c.source_id, c.external_id, c.category, c.reason,"
        " c.work_authorization, c.authorization_quote, c.fit_score,"
        " c.matched_skills, c.missing_skills "
        "FROM classifications c "
        "JOIN postings p ON p.source_id = c.source_id"
        " AND p.external_id = c.external_id "
        "LEFT JOIN notifications n ON n.source_id = c.source_id"
        " AND n.external_id = c.external_id "
        "WHERE c.prompt_version=? AND c.profile_hash=? AND c.is_match=1 "
        "AND n.source_id IS NULL "
        "ORDER BY c.created_at LIMIT ?",
        (prompt_version, profile_hash, limit),
    ).fetchall()
    return [((r["source_id"], r["external_id"]), _verdict_from_row(r))
            for r in rows]


def _verdict_from_row(row) -> Verdict:
    """Rebuild a Verdict from its stored columns.

    Skill lists are stored as a joined string, so a skill containing a comma
    splits into two here. Accepted: they are display-only, and the alternative
    is a second table for a field nothing queries.
    """
    def split(value) -> list[str]:
        return [s.strip() for s in (value or "").split(",") if s.strip()]

    return Verdict(
        id=row["external_id"],
        is_match=True,
        category=row["category"] or "",
        reason=row["reason"] or "",
        # Rows written before these columns existed hold NULL; the model's own
        # default is the right reading of "nothing was recorded either way".
        work_authorization=row["work_authorization"] or "not_mentioned",
        authorization_quote=row["authorization_quote"] or "",
        fit_score=row["fit_score"],
        matched_skills=split(row["matched_skills"]),
        missing_skills=split(row["missing_skills"]),
    )


def already_notified(conn, source_id: str, external_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM notifications WHERE source_id=? AND external_id=?",
        (source_id, external_id),
    ).fetchone() is not None


def mark_notified(conn, source_id: str, external_id: str,
                  run_id: str = "") -> None:
    """Called only after a successful send — a crash here costs one duplicate,
    which is the better failure than a permanent silent drop."""
    conn.execute(
        "INSERT OR REPLACE INTO notifications (source_id, external_id, sent_at,"
        " run_id) VALUES (?,?,?,?)",
        (source_id, external_id, now(), run_id))
    conn.commit()


# --- source health --------------------------------------------------------

def record_source_result(conn, source_id: str, ok: bool, count: int = 0,
                         quarantine_days: int = 7, threshold: int = 5) -> None:
    row = conn.execute("SELECT * FROM source_health WHERE source_id=?",
                       (source_id,)).fetchone()
    fails = (row["consecutive_fails"] if row else 0)
    if ok:
        conn.execute(
            "INSERT OR REPLACE INTO source_health VALUES (?,?,?,?,?)",
            (source_id, 0, None, now(), count),
        )
    else:
        fails += 1
        quarantine = None
        if fails >= threshold:
            quarantine = (datetime.now(UTC)
                          + timedelta(days=quarantine_days)).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO source_health VALUES (?,?,?,?,?)",
            (source_id, fails, quarantine,
             row["last_success"] if row else None,
             row["last_count"] if row else 0),
        )
    conn.commit()


def is_quarantined(conn, source_id: str) -> bool:
    row = conn.execute(
        "SELECT quarantined_until FROM source_health WHERE source_id=?",
        (source_id,)).fetchone()
    if not row or not row["quarantined_until"]:
        return False
    return datetime.fromisoformat(row["quarantined_until"]) > datetime.now(UTC)


def previous_count(conn, source_id: str) -> int | None:
    row = conn.execute("SELECT last_count FROM source_health WHERE source_id=?",
                       (source_id,)).fetchone()
    return row["last_count"] if row else None


# --- tracing --------------------------------------------------------------

def record_llm_call(conn, run_id: str, stage: str, model: str, provider: str,
                    attempt: int, status: str, http_status: int | None,
                    latency_ms: int, batch_size: int, parsed_ok: bool,
                    excerpt: str = "", session_id: str = "",
                    generation_id: str = "") -> None:
    conn.execute(
        "INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, stage, model, provider, attempt, status, http_status,
         latency_ms, batch_size, int(parsed_ok), excerpt[:2000], now(),
         session_id, generation_id),
    )
    conn.commit()


def record_error(conn, run_id: str, source_id: str | None, stage: str,
                 exc_type: str, message: str, tb: str) -> None:
    conn.execute("INSERT INTO errors VALUES (?,?,?,?,?,?,?)",
                 (run_id, source_id, stage, exc_type, message[:1000],
                  tb[:4000], now()))
    conn.commit()


def start_run(conn, run_id: str, profile_hash: str, prompt_version: str,
              mode: str = "live", session_id: str = "") -> None:
    conn.execute(
        "INSERT OR REPLACE INTO runs (run_id, started_at, profile_hash,"
        " prompt_version, mode, session_id) VALUES (?,?,?,?,?,?)",
        (run_id, now(), profile_hash, prompt_version, mode, session_id))
    conn.commit()


def finish_run(conn, run_id: str, **stats) -> None:
    fields = ", ".join(f"{k}=?" for k in stats)
    conn.execute(f"UPDATE runs SET finished_at=?, {fields} WHERE run_id=?",
                 (now(), *stats.values(), run_id))
    conn.commit()


def get_meta(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))
    conn.commit()


def hours_since(conn, key: str) -> float | None:
    """Hours since `key` was last stamped, or None if it never was."""
    raw = get_meta(conn, key)
    if not raw:
        return None
    try:
        then = datetime.fromisoformat(raw)
    except ValueError:
        return None
    # SQLite's own datetime() produces naive strings. Treat those as UTC rather
    # than raising: this runs at the end of a run, and blowing up here would
    # skip pruning and the final summary over a timestamp format.
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return (datetime.now(UTC) - then).total_seconds() / 3600


def stamp(conn, key: str) -> None:
    set_meta(conn, key, now())


def prune(conn, days: int = 90, posting_days: int = 90) -> dict:
    """Trim observability tables, and postings that have left every board.

    notifications is never touched. It is the exactly-once guarantee and it is
    tiny — one short row per job ever sent. Keeping a notification whose posting
    has been pruned is exactly the point: if that job is reposted under the same
    id, already_notified() still suppresses it.

    classifications are pruned alongside their posting, since the key they hang
    off is gone; they are re-derivable from a single LLM call if it returns.
    """
    cutoff = f"datetime('now','-{days} days')"
    conn.execute(f"DELETE FROM llm_calls WHERE created_at < {cutoff}")
    conn.execute(f"DELETE FROM errors WHERE occurred_at < {cutoff}")
    conn.execute(f"DELETE FROM runs WHERE started_at < {cutoff}")

    # last_seen is refreshed on every run a posting is still listed, so an old
    # value means the job is gone from the board, not that it is merely old.
    stale = f"datetime('now','-{posting_days} days')"
    dead = conn.execute(
        f"SELECT source_id, external_id FROM postings WHERE last_seen < {stale}"
    ).fetchall()
    for row in dead:
        conn.execute("DELETE FROM classifications WHERE source_id=? AND external_id=?",
                     (row["source_id"], row["external_id"]))
    conn.execute(f"DELETE FROM postings WHERE last_seen < {stale}")
    conn.commit()
    return {"postings_pruned": len(dead)}


def sample_rejections(conn, limit: int = 20) -> list[sqlite3.Row]:
    """False negatives are invisible by construction — read these periodically."""
    return conn.execute(
        "SELECT c.reason, c.model, p.title, p.location, p.url "
        "FROM classifications c JOIN postings p USING (source_id, external_id) "
        "WHERE c.is_match=0 ORDER BY RANDOM() LIMIT ?", (limit,)).fetchall()
