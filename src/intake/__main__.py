"""Interactive intake: run once, on your laptop, to produce config/profile.yaml.

    python -m src.intake

Strictly separate from the monitor — a scheduled job must never wait on stdin.
"""

import sys
from pathlib import Path

from pydantic import ValidationError

from src.config import PROFILE_PATH, load_env, profile_hash
from src.intake import clarify, confirm
from src.intake.extract import extract, to_profile


def attach_resume(profile) -> None:
    """Optional step: parse a resume PDF so verdicts carry a fit score.

    Skippable, and failure is never fatal — a broken PDF must not cost the
    profile that was just built over several LLM calls.
    """
    from src.intake.resume import extract_text, parse

    print("\nOptional: a resume PDF, so each match is scored against your"
          " skills.")
    print("(Enter to skip — you can add one later with"
          " `python tools/import_resume.py`)")
    try:
        raw = input("  path to PDF > ").strip().strip('"').strip("'").strip()
    except EOFError:
        return
    if not raw:
        return

    path = Path(raw).expanduser()
    try:
        text = extract_text(path)
        print("  parsing...")
        profile.resume = parse(text, source_file=path.name)
    except Exception as e:
        # Broad on purpose: a bad PDF, a rate limit or a model returning junk
        # must not cost the profile that was just built over several LLM calls.
        # The resume is optional; the profile is the point of this command.
        print(f"  skipping the resume: {e}", file=sys.stderr)
        return
    print(f"  parsed: {len(profile.resume.skills)} skills, "
          f"{len(profile.resume.tools)} technologies, "
          f"{len(profile.resume.experience)} roles")


def main() -> int:
    load_env()
    print("Describe the jobs you're looking for — role, location, timing, anything.")
    print("(blank line to finish)\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)

    text = "\n".join(lines).strip()
    if not text:
        print("nothing entered", file=sys.stderr)
        return 1

    print("\nreading that...")
    draft = extract(text)

    questions = clarify.ask(draft)
    if questions:
        print("\nA few questions (press enter to skip any):\n")
        qa = []
        for q in questions:
            try:
                qa.append((q, input(f"  {q}\n  > ")))
            except EOFError:
                break
        if any(a.strip() for _, a in qa):
            print("\nupdating...")
            draft = clarify.revise(draft, qa)

    try:
        profile = to_profile(draft)
    except ValidationError as e:
        print(f"the model produced an invalid profile:\n{e}", file=sys.stderr)
        return 1

    attach_resume(profile)

    while True:
        confirm.show(profile)
        choice = input("[a]ccept, [e]dit, [q]uit? ").strip().lower()
        if choice.startswith("a"):
            break
        if choice.startswith("e"):
            try:
                profile = confirm.edit(profile)
            except Exception as e:
                # Broad on purpose: a failed hand edit — invalid YAML, a field
                # the schema rejects, an editor that could not be launched —
                # must return to the menu, never discard the profile.
                print(f"could not use that edit: {e}")
            continue
        if choice.startswith("q"):
            print("nothing saved")
            return 1

    confirm.save(profile, PROFILE_PATH)
    print(f"\nwrote {PROFILE_PATH}")
    print(f"profile_hash: {profile_hash(profile)}")
    print("\nnext: python -m src.run --seed   (then drop --seed on later runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
