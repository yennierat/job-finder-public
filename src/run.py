"""Orchestration only — no business logic lives here.

    python -m src.run                 # normal run
    python -m src.run --seed          # populate state without notifying
    python -m src.run --dry-run       # classify + print, record nothing
    python -m src.run --test-profile  # classify a stored sample, show matches
"""

import argparse
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from src import store
from src.classify import PROMPT_VERSION, classify
from src.config import load_env, load_profile, load_sources, profile_hash
from src.enrich import enrich
from src.fetchers import fetch
from src.llm import SESSION_ID
from src.models import Posting
from src.notify import fit_band, make_notifier
from src.observability import (exit_code, health_report, log, log_error,
                               send_ops_messages)
from src.prefilter import Prefilter


# The workflow allows 30 minutes for the whole run. Fetching 98 boards takes
# ~6, enrichment ~2, and the state push a few seconds; 15 minutes of
# classification leaves genuine margin. Free models are slow enough that this
# matters: one batch has been observed taking four minutes to fail.
CLASSIFY_BUDGET_SECONDS = 15 * 60


def fetch_all(sources, conn, run_id: str, workers: int = 8):
    """Fetch every source in parallel, isolating failures per source."""
    postings, ok, failed = [], 0, 0

    def one(source):
        got = fetch(source.platform, source.id, source.board)
        if source.default_location:
            # Only fills gaps: a posting whose location did canonicalise keeps it.
            for p in got:
                if p.location is None:
                    p.location = source.default_location
        return source, got

    live = [s for s in sources if not store.is_quarantined(conn, s.id)]
    skipped = len(sources) - len(live)
    if skipped:
        log("sources.quarantined", count=skipped)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Keep the source attached to its future: without this a failure cannot
        # say which board broke, and the failure is never recorded, so the
        # circuit breaker never counts and never quarantines anything.
        futures = {pool.submit(one, s): s for s in live}
        for future in as_completed(futures):
            source = futures[future]
            try:
                _, got = future.result()
            except Exception as e:
                failed += 1
                log_error("fetch.failed", e, source=source.id)
                store.record_source_result(conn, source.id, False)
                store.record_error(conn, run_id, source.id, "fetch",
                                   type(e).__name__, str(e), traceback.format_exc())
                continue

            ok += 1
            postings.extend(got)

            # A board returning 200 with zero jobs is indistinguishable from a
            # dead tenant — nothing errored, so the breaker will never catch it.
            #
            # Read the previous count BEFORE recording this one. The success
            # path of record_source_result overwrites last_count, so reading
            # afterwards returns the value just written: `before` came back
            # equal to len(got) every time, which made the comparison below
            # `0 > 0` on exactly the runs it existed to catch.
            before = store.previous_count(conn, source.id)
            store.record_source_result(conn, source.id, True, len(got))
            if before and len(got) == 0:
                log("source.went_empty", source=source.id, previous=before)

    return postings, ok, failed


def deliver(notifier, to_send, conn, run_id: str) -> int:
    """Send each (posting, verdict) at most once. Returns the number delivered.

    There are two distinct failures here, and only one of them looks like one:

      * send() RAISES — a bug, or requests escaping its own retry loop.
      * send() returns False — the notifier tried and was refused. Telegram
        reports this after three transport failures, three 429s, or any
        non-400 status, and only once the plain-text fallback has failed too.

    The second is the dangerous one. No exception, no traceback, and by the
    time it happens the verdict is settled — the posting is no longer new, and
    `unclassified` excludes a confirmed match — so nothing downstream would
    ever look at it again. It used to fall straight through this loop and the
    job was lost in silence. Both cases are now logged and recorded, and
    store.undelivered_matches offers the posting again on the next run.
    """
    sent = 0
    for posting, verdict in to_send:
        if store.already_notified(conn, posting.source_id, posting.external_id):
            continue
        try:
            delivered = notifier.send(posting, verdict)
        except Exception as e:
            log_error("notify.failed", e, external_id=posting.external_id)
            store.record_error(conn, run_id, posting.source_id, "notify",
                               type(e).__name__, str(e), traceback.format_exc())
            continue

        if delivered:
            # Recorded only after a confirmed send, so a crash here costs one
            # duplicate rather than a permanent silent drop.
            store.mark_notified(conn, posting.source_id, posting.external_id,
                                run_id=run_id)
            sent += 1
        else:
            log("notify.refused", external_id=posting.external_id,
                source=posting.source_id)
            store.record_error(conn, run_id, posting.source_id, "notify",
                               "SendRefused",
                               "notifier returned False; queued for retry", "")
    return sent


def select_pending(conn, candidates: list[Posting], postings: list[Posting],
                   profile_hash: str) -> list[Posting]:
    """Everything this run should classify, marking what it owes before it starts.

    Three groups, in order:
      1. new postings that survived the prefilter and have no settled verdict
      2. postings an earlier run owed a verdict and never delivered one
      3. rejections still waiting for the second opinion that would confirm them

    Only the first is "new". The other two were stored by an earlier run, so
    upsert_postings will never hand them back again and they have to be pulled
    in by key. Both are resolved against what was fetched this run rather than
    against the database, which keeps the advert text available — a verdict
    formed on a bare job title is worse evidence than the one it replaces — and
    skips anything that has since left its board.
    """
    pending = store.unclassified(conn, candidates, PROMPT_VERSION, profile_hash)
    # Marked before a single model call is spent. These postings are already
    # stored, so they will never count as new again: if this run ends without a
    # verdict for one, this mark is the only thing that brings it back.
    store.mark_awaiting_verdict(conn, pending)

    queued = {(p.source_id, p.external_id) for p in pending}
    available = {(p.source_id, p.external_id): p for p in postings}

    owed = [available[key] for key in store.awaiting_verdict(conn)
            if key in available and key not in queued]
    if owed:
        log("classify.owed", count=len(owed))
        queued.update((p.source_id, p.external_id) for p in owed)

    retries = [available[key] for key
               in store.awaiting_second_opinion(conn, PROMPT_VERSION, profile_hash)
               if key in available and key not in queued]
    if retries:
        log("reclassify.pending", count=len(retries))

    # These three streams are keyed by (source_id, external_id), but classify()
    # returns verdicts keyed by external_id alone, and ids are only unique
    # within a board — two Workday tenants both number a job R-12345. Sent in
    # one run, the model answers that id once and both postings would take the
    # verdict, so a match on one board would notify for an unrelated job on
    # another. Keep the first and drop the rest: they stay marked as owed, or
    # still hold an unconfirmed rejection, so they come back next run.
    out, seen = [], set()
    for p in pending + owed + retries:
        if p.external_id in seen:
            log("classify.id_collision", external_id=p.external_id,
                source=p.source_id)
            continue
        seen.add(p.external_id)
        out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store_true",
                        help="record state without notifying (use on first run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="classify and print; persist nothing")
    parser.add_argument("--test-profile", action="store_true",
                        help="classify a stored sample to tune the profile")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    load_env()
    run_id = uuid.uuid4().hex[:12]
    profile = load_profile()
    phash = profile_hash(profile)
    conn = store.connect()
    mode = ("test" if args.test_profile else "seed" if args.seed
            else "dry-run" if args.dry_run else "live")
    store.start_run(conn, run_id, phash, PROMPT_VERSION, mode=mode,
                    session_id=SESSION_ID)
    log("run.start", run_id=run_id, profile_hash=phash,
        prompt_version=PROMPT_VERSION, mode=mode, session_id=SESSION_ID)

    prefilter = Prefilter(profile)

    if args.test_profile:
        # Classify a fixed sample of stored postings — tunes the profile without
        # thrashing the cache or the rate limit.
        rows = conn.execute(
            "SELECT source_id, external_id, title, url, location, location_raw, "
            "employment_type, remote FROM postings ORDER BY RANDOM()").fetchall()
        stored = [Posting(source_id=r["source_id"], external_id=r["external_id"],
                          title=r["title"], url=r["url"], location=r["location"],
                          location_raw=r["location_raw"] or r["location"],
                          employment_type=r["employment_type"],
                          remote=bool(r["remote"])) for r in rows]
        # Prefilter first, then cap: --limit is a budget of postings to actually
        # classify, so it is not spent on rows the prefilter would discard.
        eligible = prefilter.apply(stored)
        candidates = eligible[:args.limit]
        log("test_profile.prefiltered", stored=len(stored), eligible=len(eligible),
            classifying=len(candidates))
        verdicts = classify(candidates, profile, run_id=run_id, conn=conn)
        matched = [(p, verdicts[p.external_id][0]) for p in candidates
                   if p.external_id in verdicts and verdicts[p.external_id][0].is_match]
        matched.sort(key=lambda x: -(x[1].fit_score if x[1].fit_score
                                     is not None else -1))
        for p, verdict in matched:
            fit = ("" if verdict.fit_score is None
                   else f" — {fit_band(verdict.fit_score)[1]}")
            print(f"\n{p.title}{fit}"
                  f"\n  {p.source_id} — {p.location or p.location_raw or '?'}"
                  f"\n  {verdict.reason}"
                  f"\n  {p.url or '(no url)'}")
        print(f"\n{len(matched)} matches of {len(candidates)} classified")
        return 0

    sources = load_sources(tags=profile.source_tags)
    log("sources.loaded", count=len(sources))

    postings, ok, failed = fetch_all(sources, conn, run_id)
    log("fetch.done", postings=len(postings), sources_ok=ok, sources_failed=failed)

    new = postings if args.dry_run else store.upsert_postings(conn, postings)
    log("dedupe.done", new=len(new))

    if args.seed:
        # Deliberately skip classification: seeded postings are never notified,
        # so classifying them would spend the rate limit on verdicts nobody reads.
        store.finish_run(conn, run_id, sources_ok=ok, sources_failed=failed,
                         postings_seen=len(postings), postings_new=len(new),
                         llm_calls=0, llm_failures=0, notifications_sent=0)
        log("run.done", run_id=run_id, seeded=len(new),
            note="baseline recorded; future runs notify on new postings only")
        return exit_code(ok, failed)

    candidates = prefilter.apply(new)
    log("prefilter.done", kept=len(candidates), dropped=len(new) - len(candidates))

    pending = (candidates if args.dry_run
               else select_pending(conn, candidates, postings, phash))

    # Only now, on the handful that will actually be classified: fetch the advert
    # text for platforms that withhold it, and pull out the closing date. Doing
    # this before the prefilter would mean a request per posting across 38,000.
    if pending:
        enrich(pending, sources)
        if not args.dry_run:
            store.save_deadlines(conn, pending)
    to_send = []

    def save(batch, got):
        # Recorded as each batch returns, not once classify() has finished.
        # Held to the end, a crash or a kill by the job timeout threw away every
        # verdict the run had already paid for. A posting with no verdict writes
        # no row and keeps its awaiting_verdict mark, so the next run retries it.
        #
        # Nothing here may raise. classify() calls this from inside a try that
        # catches only AllModelsFailed, so an escaping exception — a locked
        # database, most likely — would abandon every remaining batch and skip
        # the run's own bookkeeping, which is the opposite of the point.
        for p in batch:
            if p.external_id not in got:
                continue
            verdict, meta = got[p.external_id]
            try:
                if not args.dry_run:
                    store.record_classification(conn, p, verdict, meta,
                                                PROMPT_VERSION, phash,
                                                run_id=run_id)
            except Exception as e:
                log_error("classify.save_failed", e, external_id=p.external_id)
                continue  # still marked as owed, so the next run retries it
            if verdict.is_match:
                to_send.append((p, verdict))

    # Leave headroom inside the workflow's 30-minute limit for fetching,
    # enrichment and the state push. Being killed by the job timeout loses the
    # run's own bookkeeping; stopping early only defers work to the next run.
    verdicts = classify(pending, profile, run_id=run_id, conn=conn,
                        budget_seconds=CLASSIFY_BUDGET_SECONDS,
                        on_batch=save) if pending else {}
    log("classify.done", requested=len(pending), returned=len(verdicts))

    notifier = make_notifier()
    log("notifier.selected", kind=type(notifier).__name__)

    sent = 0
    if not args.dry_run:   # --seed returned above; --dry-run persists nothing
        # Matches from earlier runs that were never delivered. Their verdicts
        # are settled, so no other stage will ever reconsider them — this is
        # the only thing standing between a refused send and a lost job.
        # Resolved against what was fetched this run, so a message is only sent
        # for a posting still open on its board.
        queued = {(p.source_id, p.external_id) for p, _ in to_send}
        available = {(p.source_id, p.external_id): p for p in postings}
        redeliver = [(available[key], verdict) for key, verdict
                     in store.undelivered_matches(conn, PROMPT_VERSION, phash)
                     if key in available and key not in queued]
        if redeliver:
            log("notify.redelivering", count=len(redeliver))
        sent = deliver(notifier, to_send + redeliver, conn, run_id)

    llm_stats = conn.execute(
        "SELECT COUNT(*) n, SUM(status!='ok') f FROM llm_calls WHERE run_id=?",
        (run_id,)).fetchone()

    store.finish_run(conn, run_id, sources_ok=ok, sources_failed=failed,
                     postings_seen=len(postings), postings_new=len(new),
                     llm_calls=llm_stats["n"] or 0,
                     llm_failures=llm_stats["f"] or 0, notifications_sent=sent)
    ops = send_ops_messages(conn, notifier, ok, failed,
                            llm_failures=llm_stats["f"] or 0,
                            verdicts_requested=len(pending),
                            verdicts_returned=len(verdicts))
    pruned = store.prune(conn)
    log("run.done", run_id=run_id, notifications_sent=sent,
        health=health_report(conn), **ops, **pruned)

    # A run that fetched postings and judged none of them has not succeeded,
    # whatever the boards did. Failing here is what fires the workflow's own
    # alert, which does not sit behind a cooldown.
    return exit_code(ok, failed, len(pending), len(verdicts))


if __name__ == "__main__":
    raise SystemExit(main())
