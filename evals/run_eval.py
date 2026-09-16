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
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.classify import BATCH_SIZE, PROMPT_VERSION, classify  # noqa: E402
from src.config import load_env, load_profile  # noqa: E402
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
    args = parser.parse_args()

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
    results = defaultdict(list)   # id -> [(ok, detail, score)]

    for run in range(args.repeats):
        verdicts = classify(postings, profile, run_id=f"eval-{run}",
                            batch_size=BATCH_SIZE)
        for case in cases:
            got = verdicts.get(case["id"])
            verdict = got[0] if got else None
            ok, detail = judge(case, verdict)
            results[case["id"]].append(
                (ok, detail, verdict.fit_score if verdict else None))
        done = sum(1 for c in cases if results[c["id"]][-1][0])
        print(f"  repeat {run + 1}: {done}/{len(cases)}", flush=True)

    print()
    header = f"{'rate':>6}  {'scores':>14}  case"
    print(header)
    print("-" * (len(header) + 24))

    failures, unstable = [], []
    for case in cases:
        runs = results[case["id"]]
        passed = sum(1 for ok, _, _ in runs if ok)
        rate = passed / len(runs)
        scores = [s for _, _, s in runs if s is not None]
        spread = (f"{min(scores)}-{max(scores)}" if scores else "-")
        if len(scores) > 1 and len(set(scores)) > 1:
            spread += f" ±{int(statistics.pstdev(scores))}"

        mark = "ok" if rate == 1 else ("~" if rate > 0 else "FAIL")
        known = " (known-fail)" if "known-fail" in case.get("tags", []) else ""
        print(f"{passed}/{len(runs)} {mark:<5} {spread:>14}  {case['id']}{known}")

        if rate < 1:
            reasons = {d for ok, d, _ in runs if not ok}
            print(f"                        {'; '.join(sorted(reasons))}")
            print(f"                        expected: {case['why'].strip()}")
            (failures if rate == 0 else unstable).append(case["id"])

    # `if ok` is load-bearing: without it this counts every judgement rather
    # than every passing one, reports 100% regardless, and the threshold below
    # can never fire. An eval that cannot fail is worse than no eval — it is a
    # green light wired to nothing.
    total = sum(sum(1 for ok, _, _ in r if ok) for r in results.values())
    possible = len(cases) * args.repeats
    overall = total / possible

    print(f"\noverall {total}/{possible} = {overall:.0%}")
    if unstable:
        # Not a separate bug — the model's own variance, surfacing. Worth
        # seeing, because a case that flips is a case near the prompt's edge.
        print(f"unstable (passed some repeats, not all): {', '.join(unstable)}")
    if failures:
        print(f"failing every repeat: {', '.join(failures)}")

    if overall < args.threshold:
        print(f"\nBELOW THRESHOLD ({args.threshold:.0%})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
