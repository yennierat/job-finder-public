"""Classify every posting that passes your profile's prefilter, and save the matches.

Unlike `src.run`, this persists nothing to state.db and sends no Telegram — it
just writes a shortlist you can work through. Safe to run repeatedly.

    python tools/shortlist.py              # use stored postings (fast)
    python tools/shortlist.py --fresh      # re-fetch first, so descriptions are
                                           # available and visa clauses get read
    python tools/shortlist.py --limit 50   # cap the number classified

Output: shortlist.md (readable), shortlist.json (raw) and shortlist.csv
(tracking — has a `status` column to fill in as you apply).

shortlist.csv is written only if it does not already exist, so re-running never
overwrites the statuses you have typed. New matches are appended.

Note on --fresh: descriptions are not stored in state.db (they would add ~25MB),
so without it the classifier sees only title, location and type, and
work_authorization is always "not_mentioned".
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import store  # noqa: E402
from src.classify import BATCH_SIZE, classify  # noqa: E402
from src.config import load_env, load_profile, load_sources  # noqa: E402
from src.enrich import enrich  # noqa: E402
from src.models import Posting  # noqa: E402
from src.normalise import days_until  # noqa: E402
from src.notify import AUTHORIZATION, company_of, deadline_line, fit_band  # noqa: E402
from src.prefilter import Prefilter  # noqa: E402


def authz_mark(value: str) -> tuple[str, str]:
    """(short badge, prose label) for the shortlist, from notify's one table."""
    entry = AUTHORIZATION.get(value)
    return (entry[2], entry[1]) if entry else ("", "")


def from_db(conn) -> list:
    rows = conn.execute(
        "SELECT source_id, external_id, title, url, location, location_raw,"
        " employment_type, remote FROM postings").fetchall()
    return [Posting(source_id=r["source_id"], external_id=r["external_id"],
                    title=r["title"], url=r["url"], location=r["location"],
                    location_raw=r["location_raw"] or r["location"],
                    employment_type=r["employment_type"],
                    remote=bool(r["remote"])) for r in rows]


def from_network(profile) -> list:
    """Re-fetch so descriptions (and therefore visa clauses) are available."""
    from concurrent.futures import ThreadPoolExecutor

    from src.fetchers import fetch

    sources = load_sources(tags=profile.source_tags)

    def one(s):
        try:
            got = fetch(s.platform, s.id, s.board)
        except Exception as e:
            tqdm.write(f"  ! {s.id}: {type(e).__name__}")
            return []
        if s.default_location:
            for p in got:
                if p.location is None:
                    p.location = s.default_location
        return got

    postings = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for got in tqdm(pool.map(one, sources), total=len(sources),
                        desc="fetching boards", unit="board"):
            postings.extend(got)
    return postings


def render_markdown(matches: list, total_classified: int) -> str:
    by_place = defaultdict(list)
    for p, v in matches:
        by_place[p.location or p.location_raw or "Location unknown"].append((p, v))

    lines = [f"# Shortlist — {date.today().isoformat()}", "",
             f"{len(matches)} matches out of {total_classified} classified.", ""]

    # Anything closing within a fortnight goes at the top, regardless of fit or
    # location. A strong match found four days late is worth nothing.
    urgent = sorted(((p, v) for p, v in matches
                     if (d := days_until(p.deadline)) is not None and 0 <= d <= 14),
                    key=lambda x: x[0].deadline)
    if urgent:
        lines += ["## ⏰ Closing within 14 days", ""]
        for p, v in urgent:
            lines.append(f"- [ ] **{p.title}** — {company_of(p.source_id)}"
                         f" — {deadline_line(p)}")
            if p.url:
                lines.append(f"      {p.url}")
        lines.append("")

    for place in sorted(by_place, key=lambda k: (-len(by_place[k]), k)):
        lines.append(f"## {place}")
        lines.append("")
        # Best fit first where scores exist, alphabetical where they do not.
        # Unscored postings sort last rather than as zero: no resume configured
        # is not the same as a bad fit, and burying them would hide matches.
        for p, v in sorted(by_place[place],
                           key=lambda x: (-(x[1].fit_score if x[1].fit_score
                                            is not None else -1), x[0].title)):
            mark = authz_mark(v.work_authorization)[0]
            company = company_of(p.source_id)
            head = f"- [ ] **{p.title}** — {company}"
            if v.fit_score is not None:
                head += f" — **{fit_band(v.fit_score)[1]}**"
            if mark:
                head += f" `{mark}`"
            closing = deadline_line(p)
            if closing:
                head += f" — _{closing}_"
            lines.append(head)
            if v.reason:
                lines.append(f"      {v.reason}")
            if v.matched_skills or v.missing_skills:
                bits = []
                if v.matched_skills:
                    bits.append("has " + ", ".join(v.matched_skills))
                if v.missing_skills:
                    bits.append("lacks " + ", ".join(v.missing_skills))
                lines.append(f"      _{' · '.join(bits)}_")
            if v.authorization_quote:
                lines.append(f'      > "{v.authorization_quote.strip()}"')
            if p.url:
                lines.append(f"      {p.url}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true",
                        help="re-fetch boards so descriptions are available")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap postings classified (0 = no cap)")
    parser.add_argument("--min-fit", type=int, default=0,
                        help="drop matches scoring below this (needs a resume; "
                             "unscored matches are always kept)")
    args = parser.parse_args()

    load_env()
    profile = load_profile()
    conn = store.connect()

    if profile.resume and not profile.resume.is_empty():
        print(f"scoring against resume: {profile.resume.source_file or 'stored'}"
              f" ({len(profile.resume.skills)} skills,"
              f" {len(profile.resume.tools)} technologies)")
    elif args.min_fit:
        print("--min-fit has no effect: no resume in config/profile.yaml "
              "(add one with `python tools/import_resume.py`)")

    postings = from_network(profile) if args.fresh else from_db(conn)
    candidates = Prefilter(profile).apply(postings)
    print(f"{len(postings)} postings -> {len(candidates)} pass your prefilter")
    if args.limit:
        candidates = candidates[:args.limit]
        print(f"capped to {len(candidates)}")
    if not candidates:
        print("nothing to classify")
        return 0

    # Advert text and closing dates, for the survivors only.
    stats = enrich(candidates, load_sources(tags=profile.source_tags))
    print(f"fetched {stats['fetched']} descriptions "
          f"({stats['failed']} refused); {stats['deadlines']} deadlines found")

    batches = [candidates[i:i + BATCH_SIZE]
               for i in range(0, len(candidates), BATCH_SIZE)]
    matches = []

    # Batch by batch rather than one call, so progress is visible and a mid-way
    # rate limit does not discard everything already judged.
    bar = tqdm(batches, desc="classifying", unit="batch")
    for batch in bar:
        verdicts = classify(batch, profile, run_id="shortlist", conn=conn)
        for p in batch:
            got = verdicts.get(p.external_id)
            if not got or not got[0].is_match:
                continue
            v = got[0]
            # Unscored verdicts survive any threshold: with no resume configured
            # every score is None, and filtering on that would empty the list.
            if v.fit_score is not None and v.fit_score < args.min_fit:
                continue
            matches.append((p, v))
            mark = authz_mark(v.work_authorization)[1]
            fit = f"{fit_band(v.fit_score)[2]:9}" if v.fit_score is not None else ""
            tqdm.write(f"  MATCH  {fit}{p.title[:50]:52} {p.location or '?':16}"
                       + (f" ({mark})" if mark else ""))
        bar.set_postfix(matches=len(matches))

    (ROOT / "shortlist.md").write_text(
        render_markdown(matches, len(candidates)), encoding="utf-8")
    (ROOT / "shortlist.json").write_text(json.dumps([
        {"title": p.title, "company": company_of(p.source_id),
         "location": p.location or p.location_raw, "url": p.url,
         "employment_type": p.employment_type, "reason": v.reason,
         "work_authorization": v.work_authorization,
         "authorization_quote": v.authorization_quote,
         "fit_score": v.fit_score, "matched_skills": v.matched_skills,
         "missing_skills": v.missing_skills,
         "deadline": p.deadline, "deadline_text": p.deadline_text}
        for p, v in matches], indent=2), encoding="utf-8")

    added = write_csv(matches)
    print()
    print(f"{len(matches)} matches -> shortlist.md, shortlist.json, "
          f"shortlist.csv ({added} new rows)")
    return 0


CSV_FIELDS = ["status", "fit", "deadline", "company", "title", "location",
              "work_authorization", "url", "applied_on", "notes"]


def write_csv(matches: list) -> int:
    """Append new matches, preserving any status already recorded.

    Keyed on url: re-running must never wipe the fact that you already applied
    to something, which is what rewriting the file wholesale would do.
    """
    path = ROOT / "shortlist.csv"
    existing: dict[str, dict] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            existing = {r["url"]: r for r in csv.DictReader(f) if r.get("url")}

    added = 0
    for p, v in matches:
        if not p.url or p.url in existing:
            continue
        existing[p.url] = {
            "status": "todo", "deadline": p.deadline or "",
            "fit": "" if v.fit_score is None else fit_band(v.fit_score)[2],
            "company": company_of(p.source_id),
            "title": p.title,
            "location": p.location or p.location_raw or "",
            "work_authorization": v.work_authorization,
            "url": p.url, "applied_on": "", "notes": v.reason,
        }
        added += 1

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        # todo first, then soonest deadline, then band, then company. A dated
        # row outranks an undated one: it is the only row that can expire.
        w.writerows(sorted(existing.values(),
                           key=lambda r: (r.get("status") != "todo",
                                          r.get("deadline") or "9999-12-31",
                                          _band_rank(r.get("fit")),
                                          r.get("company", ""))))
    return added


# Rows written before the band change hold a bare number, and the file is edited
# by hand besides, so both forms have to sort.
_BAND_RANK = {"strong": 0, "possible": 1, "stretch": 2}


def _band_rank(value) -> int:
    text = str(value or "").strip().lower()
    if text in _BAND_RANK:
        return _BAND_RANK[text]
    try:
        return _BAND_RANK[fit_band(int(text.rstrip("%")))[2]]
    except (TypeError, ValueError):
        return len(_BAND_RANK)  # unscored sorts last, never as a bad fit


if __name__ == "__main__":
    raise SystemExit(main())
