"""Parse a resume PDF once and store the summary in config/profile.yaml.

    python tools/import_resume.py path/to/cv.pdf
    python tools/import_resume.py            # prompts for the path
    python tools/import_resume.py cv.pdf --dry-run   # show it, save nothing
    python tools/import_resume.py --remove           # drop the stored resume

Run this once. Thereafter every classification scores postings against the
stored summary and reports a fit percentage; the PDF itself is never read again
and never needs to be committed.

Two consequences worth knowing before running it:

  * The profile hash changes, because the resume is part of the profile. Cached
    verdicts are keyed on that hash, so the next run reclassifies everything —
    which is correct, since verdicts now depend on the resume. Nothing is
    re-notified: the notifications table is keyed on the posting, not the hash.

  * config/profile.yaml gains personal data. It is already in a private repo;
    keep it that way, and see --remove before making anything public.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PROFILE_PATH, load_env, load_profile, profile_hash  # noqa: E402
from src.intake import confirm  # noqa: E402
from src.intake.resume import ResumeError, extract_text, parse  # noqa: E402
from src.models import Resume  # noqa: E402


def ask_for_path() -> Path | None:
    """Prompt for a file, tolerating what a terminal actually gives you.

    Dragging a file into a Windows or macOS terminal pastes it quoted, and often
    with a trailing space. Failing on that would be an unnecessary lesson in
    shell quoting.
    """
    print("Path to your resume PDF (drag the file into this window, then Enter):")
    try:
        raw = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    raw = raw.strip().strip('"').strip("'").strip()
    return Path(raw).expanduser() if raw else None


def show(resume: Resume) -> None:
    print("\n" + "=" * 60)
    print(confirm.to_yaml_obj(resume))
    print("=" * 60)
    print("Check this before accepting. Anything wrong here is wrong in every")
    print("fit score from now on, and a hallucinated skill is invisible later.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", nargs="?", help="path to the resume PDF")
    parser.add_argument("--dry-run", action="store_true",
                        help="parse and print; do not touch profile.yaml")
    parser.add_argument("--remove", action="store_true",
                        help="delete the stored resume from profile.yaml")
    parser.add_argument("--text-only", action="store_true",
                        help="print the extracted (redacted) text and stop; no "
                             "LLM call, for checking what would be sent")
    args = parser.parse_args()

    load_env()
    profile = load_profile()

    if args.remove:
        if profile.resume is None:
            print("no resume stored")
            return 0
        profile.resume = None
        confirm.save(profile, PROFILE_PATH)
        print(f"removed; profile_hash is now {profile_hash(profile)}")
        return 0

    path = Path(args.pdf).expanduser() if args.pdf else ask_for_path()
    if path is None:
        print("nothing to do", file=sys.stderr)
        return 1

    try:
        text = extract_text(path)
    except ResumeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.text_only:
        from src.intake.resume import redact
        print(redact(text))
        return 0

    print(f"read {len(text)} characters from {path.name}; parsing...")
    try:
        resume = parse(text, source_file=path.name)
    except Exception as e:
        print(f"error: could not parse that resume: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1

    show(resume)
    if args.dry_run:
        print("\n--dry-run: profile.yaml unchanged")
        return 0

    if profile.resume is not None:
        print(f"\nA resume is already stored ({profile.resume.source_file or '?'}"
              f", parsed {profile.resume.parsed_on}). Accepting replaces it.")

    try:
        choice = input("\n[a]ccept, [q]uit? ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "q"
    if not choice.startswith("a"):
        print("nothing saved")
        return 1

    before = profile_hash(profile)
    profile.resume = resume
    confirm.save(profile, PROFILE_PATH)
    after = profile_hash(profile)
    print(f"\nwrote {PROFILE_PATH}")
    print(f"profile_hash: {before} -> {after}")
    print("\nThe next run reclassifies everything under the new hash, and every")
    print("verdict from now on carries a fit score.")
    print("Edit config/profile.yaml by hand to correct anything above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
