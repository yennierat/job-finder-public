"""Run the labelled cases in cases.yaml against the real classifier.

    python evals/run_eval.py                  # every case, 3 repeats
    python evals/run_eval.py --repeats 5
    python evals/run_eval.py --tag regression # only cases that once shipped wrong
    python evals/run_eval.py --skip-known-fail

Deliberately NOT in tests/. It needs an API key and a network, it costs rate
limit, and it is not deterministic — putting it in the offline suite would make
the pre-push hook slow and flaky, which is how a hook becomes one you skip.

Run it before committing a prompt change. That is the whole purpose: the offline
suite could not see either of the two prompt regressions that shipped, because
it tests parsing and caching rather than judgement.

WHY REPEATS. The model is not deterministic at temperature 0 — re-judging one
posting against a byte-identical prompt was measured varying by ~15 points, and
one posting went no / yes / yes across three consecutive calls. A single run
would therefore fail at random and quickly be ignored. Each case is judged
`--repeats` times and reported as a pass RATE; a case that passes 2 of 3 is
telling you something real about stability, not about correctness.

Cases are classified in batches, the same way production does it, so a verdict
is influenced by its neighbours exactly as it is in a real run.

Each repeat gets a time budget (--round-minutes), which also caps every model
attempt the way a monitor run does. A case the models never answered is
reported as unjudged and left out of the pass rate: a slow or failing model
chain is not the classifier's judgement getting worse.

eval-log.jsonl records every model attempt (model, status, latency) and every
case result as they happen, so a run killed part-way still leaves its evidence.
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import store  # noqa: E402
from src.classify import BATCH_SIZE, PROMPT_VERSION, classify  # noqa: E402
from src.config import load_env, load_profile  # noqa: E402
from src.llm import STATUS_TEXT, describe_failures  # noqa: E402
from src.models import Posting  # noqa: E402

CASES = Path(__file__).resolve().parent / "cases.yaml"


def load_cases(path=CASES):
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases = raw.get("cases") or []
    ids = [c["id"] for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise SystemExit(f"duplicate case ids: {sorted(duplicates)}")
    return cases


def to_posting(case) -> Posting:
    return Posting(
        source_id="eval", external_id=case["id"], title=case["title"],
        location=case.get("location"), location_raw=case.get("location"),
        employment_type=case.get("employment_type", "internship"),
        description=case.get("description"),
        url=f"https://example.test/{case['id']}",
    )


def judge(case, verdict) -> tuple[bool, str]:
    """Did this verdict satisfy the case? Returns (ok, what went wrong)."""
    if verdict is None:
        return False, "no verdict returned"

    if verdict.is_match != case["match"]:
        want = "match" if case["match"] else "reject"
        got = "match" if verdict.is_match else "reject"
        return False, f"wanted {want}, got {got}"

    # Score bounds are checked only when the verdict itself was right: a wrong
    # verdict with an out-of-range score is one failure, not two.
    score = verdict.fit_score
    if score is not None:
        low, high = case.get("min_score"), case.get("max_score")
        if low is not None and score < low:
            return False, f"score {score} below min {low}"
        if high is not None and score > high:
            return False, f"score {score} above max {high}"
    return True, ""


class EvalLog:
    """eval-log.jsonl, written line by line as the run goes."""

    def __init__(self, path: Path, conn):
        self.file = open(path, "w", encoding="utf-8")
        self.conn = conn
        self.last_call = 0

    def write(self, **record) -> None:
        self.file.write(json.dumps(record, default=str) + "\n")
        self.file.flush()

    def flush_calls(self) -> None:
        """Copy model attempts recorded since the last flush."""
        rows = self.conn.execute(
            "SELECT rowid, run_id, model, provider, attempt, status, http_status,"
            " latency_ms, batch_size, raw_response_excerpt, generation_id"
            " FROM llm_calls WHERE rowid > ? ORDER BY rowid",
            (self.last_call,)).fetchall()
        for r in rows:
            self.last_call = r["rowid"]
            self.write(type="call", run=r["run_id"], model=r["model"],
                       provider=r["provider"], attempt=r["attempt"],
                       status=r["status"],
                       status_text=STATUS_TEXT.get(r["status"], r["status"]),
                       http_status=r["http_status"], latency_ms=r["latency_ms"],
                       batch_size=r["batch_size"],
                       generation_id=r["generation_id"],
                       excerpt=(r["raw_response_excerpt"] or "")[:500])

    def close(self) -> None:
        self.file.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3,
                        help="judgements per case; the model is not stable")
    parser.add_argument("--tag", action="append", default=[],
                        help="only cases carrying this tag (repeatable)")
    parser.add_argument("--skip-known-fail", action="store_true",
                        help="omit cases tagged known-fail")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="overall pass rate below which this exits non-zero")
    parser.add_argument("--round-minutes", type=float, default=18,
                        help="time budget per repeat; 0 for none")
    parser.add_argument("--total-minutes", type=float, default=0,
                        help="time budget for all repeats, split evenly; "
                             "overrides --round-minutes")
    parser.add_argument("--log", type=Path, default=Path("eval-log.jsonl"),
                        help="where to write the per-call and per-case log")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    load_env()
    profile = load_profile()
    cases = load_cases()

    if args.tag:
        wanted = set(args.tag)
        cases = [c for c in cases if wanted & set(c.get("tags", []))]
    if args.skip_known_fail:
        cases = [c for c in cases if "known-fail" not in c.get("tags", [])]
    if not cases:
        print("no cases selected")
        return 0

    print(f"{len(cases)} cases x {args.repeats} repeats | PROMPT_VERSION "
          f"{PROMPT_VERSION} | resume "
          f"{'attached' if profile.resume else 'absent'}\n", flush=True)

    postings = [to_posting(c) for c in cases]
    by_id = {c["id"]: c for c in cases}
    results = defaultdict(list)   # id -> [{ok, judged, detail, score}]
    minutes = (args.total_minutes / args.repeats if args.total_minutes
               else args.round_minutes)
    budget = minutes * 60 or None

    with tempfile.TemporaryDirectory() as tmp:
        conn = store.connect(Path(tmp) / "eval.db")
        elog = None
        try:
            elog = EvalLog(args.log, conn)
            for run in range(args.repeats):
                run_evals(run, postings, by_id, profile, conn, elog, results,
                          budget)
            call_failures = store.llm_failures_by_status(conn)
        finally:
            if elog is not None:
                elog.flush_calls()
                elog.close()
            conn.close()

    print()
    header = f"{'rate':>6}  {'scores':>14}  case"
    print(header)
    print("-" * (len(header) + 24))

    failures, unstable, never_answered = [], [], []
    for case in cases:
        runs = [r for r in results[case["id"]] if r["judged"]]
        unjudged = len(results[case["id"]]) - len(runs)
        known = " (known-fail)" if "known-fail" in case.get("tags", []) else ""
        note = f"  ({unjudged} unjudged)" if unjudged else ""
        if not runs:
            print(f"{'-':>6} {'NONE':<5} {'-':>14}  {case['id']}{known}{note}")
            never_answered.append(case["id"])
            continue
        passed = sum(1 for r in runs if r["ok"])
        rate = passed / len(runs)
        scores = [r["score"] for r in runs if r["score"] is not None]
        spread = (f"{min(scores)}-{max(scores)}" if scores else "-")
        if len(scores) > 1 and len(set(scores)) > 1:
            spread += f" ±{int(statistics.pstdev(scores))}"

        mark = "ok" if rate == 1 else ("~" if rate > 0 else "FAIL")
        print(f"{passed}/{len(runs)} {mark:<5} {spread:>14}  {case['id']}{known}{note}")

        if rate < 1:
            reasons = {r["detail"] for r in runs if not r["ok"]}
            print(f"                        {'; '.join(sorted(reasons))}")
            print(f"                        expected: {case['why'].strip()}")
            (failures if rate == 0 else unstable).append(case["id"])

    # Only judged cases count: an unanswered case says nothing about judgement.
    entries = [r for rs in results.values() for r in rs]
    judged = [r for r in entries if r["judged"]]
    passed = sum(1 for r in judged if r["ok"])
    unjudged = len(entries) - len(judged)

    if call_failures:
        print(f"\nmodel call failures {describe_failures(call_failures)}")
    if not judged:
        print("\noverall nothing judged: no model answered any case")
        return 1
    overall = passed / len(judged)
    print(f"\noverall {passed}/{len(judged)} = {overall:.0%}")
    if unjudged:
        print(f"unjudged {unjudged} of {len(entries)} (no answer from the models; "
              f"not counted)")
    if unstable:
        # Not a separate bug — the model's own variance, surfacing. Worth
        # seeing, because a case that flips is a case near the prompt's edge.
        print(f"unstable (passed some repeats, not all): {', '.join(unstable)}")
    if failures:
        print(f"failing every repeat: {', '.join(failures)}")
    if never_answered:
        print(f"never answered: {', '.join(never_answered)}")

    if unjudged > len(entries) / 2:
        print("\nTOO FEW JUDGED: most cases got no answer, so the rate proves little")
        return 1
    if never_answered:
        print("\nNEVER ANSWERED: some cases got no verdict in any repeat")
        return 1
    if overall < args.threshold:
        print(f"\nBELOW THRESHOLD ({args.threshold:.0%})")
        return 1
    return 0


def run_evals(run, postings, by_id, profile, conn, elog, results, budget):
    """One repeat: classify every case, printing and logging each as it lands."""
    run_id = f"eval-{run}"
    settled = set()
    print(f"  repeat {run + 1}:", flush=True)

    returned = set()   # ids sent in a batch the model did answer

    def settle(case_id, verdict, missing=""):
        if case_id in settled:
            return
        settled.add(case_id)
        case = by_id[case_id]
        if verdict is None and case_id in returned:
            # The model answered and skipped this one: that is its judgement
            # failing, not the chain, so it counts.
            entry = {"ok": False, "judged": True, "score": None,
                     "detail": "left out of the model's answer"}
        elif verdict is None:
            entry = {"ok": False, "judged": False, "score": None,
                     "detail": f"no verdict: {missing}"}
        else:
            ok, detail = judge(case, verdict)
            entry = {"ok": ok, "judged": True, "detail": detail,
                     "score": verdict.fit_score}
        results[case_id].append(entry)
        mark = "ok" if entry["ok"] else ("WRONG" if entry["judged"] else "--")
        print(f"    {mark:<5} {case_id}"
              f"{': ' + entry['detail'] if entry['detail'] else ''}", flush=True)
        elog.write(type="case", run=run_id, id=case_id,
                   expected="match" if case["match"] else "reject",
                   got=(None if verdict is None
                        else "match" if verdict.is_match else "reject"),
                   ok=entry["ok"], judged=entry["judged"], score=entry["score"],
                   detail=entry["detail"],
                   reason=verdict.reason if verdict is not None else None)

    def on_batch(batch, got):
        for p in batch:
            returned.add(p.external_id)
            if p.external_id in got:
                settle(p.external_id, got[p.external_id][0])
        elog.flush_calls()

    t0 = time.monotonic()
    verdicts = classify(postings, profile, run_id=run_id, conn=conn,
                        batch_size=BATCH_SIZE, budget_seconds=budget,
                        on_batch=on_batch)
    elapsed = time.monotonic() - t0
    ran_out = budget is not None and elapsed >= budget
    for case_id in by_id:
        if case_id not in verdicts:
            settle(case_id, None,
                   "time budget ran out" if ran_out else "every model failed")
    elog.flush_calls()

    round_entries = [results[i][-1] for i in by_id]
    passed = sum(1 for r in round_entries if r["ok"])
    unjudged = sum(1 for r in round_entries if not r["judged"])
    print(f"  repeat {run + 1}: {passed}/{len(by_id)} passed, {unjudged} unjudged, "
          f"{elapsed:.0f}s", flush=True)
    failed = store.llm_failures_by_status(conn, run_id)
    if failed:
        print(f"    model call failures: {describe_failures(failed)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
