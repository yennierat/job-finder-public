"""Tests for deadline extraction, enrichment and the closing-date display.

Deadlines are extracted by rule rather than by the model, so these tests carry
the whole correctness argument for the feature. Two failure modes matter, and
they are not equally bad:

  * a MISSED deadline costs a line in a message
  * a WRONG deadline is trusted and acted on, and loses the application

So the bias throughout is towards finding nothing rather than guessing, and most
of what follows checks that dates which are not deadlines are ignored.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import notify  # noqa: E402
from src.enrich import enrich  # noqa: E402
from src.models import Posting, Verdict  # noqa: E402
from src.normalise import days_until, extract_deadline  # noqa: E402

failures = []
TODAY = date(2026, 9, 5)


def check(name, got, want):
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


def ok(name, cond):
    if not cond:
        failures.append(f"{name}: expected true")


def found(text):
    return extract_deadline(text, today=TODAY)[0]


# --- formats that must be recognised --------------------------------------
# Every one of these is phrasing taken from a real advert.

for text, want in [
    ("Application Deadline: Sep 27, 2026", "2026-09-27"),
    ("Application deadline: 27 September 2026", "2026-09-27"),
    ("Applications close on 30 September 2026.", "2026-09-30"),
    ("Please apply by 15 January 2027.", "2027-01-15"),
    ("Closing date: 2026-12-31", "2026-12-31"),
    ("Last day to apply: 3rd October 2026", "2026-10-03"),
    ("Applications will close on Friday, 9 October 2026", "2026-10-09"),
    ("Submissions are due by December 1, 2026", "2026-12-01"),
    ("apply before 28 Feb 2027", "2027-02-28"),
    ("APPLICATION DEADLINE: 30 NOVEMBER 2026", "2026-11-30"),
]:
    check(f"parses {text[:38]!r}", found(text), want)


# --- dates that are NOT deadlines -----------------------------------------
# Adverts are full of dates. Taking any of them would be worse than none.

for text in [
    "The internship starts on 1 June 2027 and runs for 12 weeks.",
    "Founded in 1985, we are a global trading firm.",
    "This posting was published on 4 September 2026.",
    "You will graduate between May 2027 and August 2027.",
    "Our summer programme ran from 3 July 2025.",
    "Interviews will be held in October 2026.",
    "",
]:
    check(f"ignores {text[:44]!r}", found(text), None)

check("ignores None", found(None), None)

# A date far outside the plausible window means the trigger matched something
# that was not a deadline after all.
check("rejects a long-past deadline",
      found("Application deadline: 12 March 2019"), None)
check("rejects an implausibly distant deadline",
      found("Application deadline: 1 January 2045"), None)
# Just-passed deadlines are kept: a posting can close days before a run, and
# saying "closed 2 Sep" is more useful than showing nothing.
ok("keeps a recently passed deadline",
   found("Applications closed on 2 September 2026") == "2026-09-02")

# The trigger has to be near the date, not merely somewhere in the document.
far = ("Application deadline is stated below. " + "filler text. " * 40
       + "1 December 2026")
check("date too far from the trigger is ignored", found(far), None)

# Malformed dates must not raise.
for bad in ("Application deadline: 31 February 2027",
            "Application deadline: 45 Movember 2026",
            "Closing date: 2026-13-45"):
    check(f"survives {bad[-18:]!r}", found(bad), None)


# --- the quote makes it auditable -----------------------------------------

iso, sentence = extract_deadline(
    "We hire on a rolling basis. Applications close on 30 September 2026. "
    "Interviews follow.", today=TODAY)
check("date found", iso, "2026-09-30")
ok("sentence returned", sentence and "30 September 2026" in sentence)
# The quote is a sentence, not the whole advert: it exists to be checked at a
# glance against the source.
ok("sentence is bounded", len(sentence) <= 160)
ok("sentence excludes neighbours", "Interviews follow" not in sentence)


# --- days_until -----------------------------------------------------------

check("days until future", days_until("2026-09-15", today=TODAY), 10)
check("days until today", days_until("2026-09-05", today=TODAY), 0)
check("days since past", days_until("2026-09-01", today=TODAY), -4)
check("no date, no number", days_until(None), None)
check("garbage date, no number", days_until("not a date"), None)


# --- the message ----------------------------------------------------------

def posting(**kw):
    base = dict(source_id="imc-greenhouse", external_id="1",
                title="Software Engineer Intern", location="Singapore",
                employment_type="internship", url="https://example.com/1")
    return Posting(**{**base, **kw})


soon = (date.today() + timedelta(days=9)).isoformat()
line = notify.deadline_line(posting(deadline=soon))
ok("counts the days down", "closes in 9 days" in line)
ok("names the date too", str(date.today().year) in line or "20" in line)

check("no line without a deadline", notify.deadline_line(posting()), None)
ok("singular day",
   "closes in 1 day" in notify.deadline_line(
       posting(deadline=(date.today() + timedelta(days=1)).isoformat())))
ok("today is called out",
   "TODAY" in notify.deadline_line(posting(deadline=date.today().isoformat())))
# A passed deadline must never render as a negative countdown.
past = notify.deadline_line(
    posting(deadline=(date.today() - timedelta(days=3)).isoformat()))
ok("past reads as closed", past.startswith("closed") and "-3" not in past)
# A malformed stored value must not break a notification.
ok("garbage deadline is ignored",
   notify.deadline_line(posting(deadline="soon-ish")) is None)

v = Verdict(id="1", is_match=True, reason="Matches.", fit_score=75)
msg = notify.format_message(posting(deadline=soon), v)
ok("plain message shows the deadline", "closes in 9 days" in msg)
h = notify.format_html(posting(deadline=soon), v)
ok("html message shows the deadline", "closes in 9 days" in h)
# Under a fortnight the deadline outranks the fit band as the thing to act on.
ok("urgent deadline is bolded", "<b>closes in 9 days" in h)
far_off = (date.today() + timedelta(days=90)).isoformat()
ok("distant deadline is not bolded",
   "<b>closes in 90 days" not in notify.format_html(posting(deadline=far_off), v))
ok("no clock without a deadline", "⏰" not in notify.format_html(posting(), v))


# --- enrichment -----------------------------------------------------------

class Source:
    def __init__(self, id, platform, board="b"):
        self.id, self.platform, self.board = id, platform, board


sources = [Source("wd", "workday"), Source("gh", "greenhouse")]

import src.enrich as enrich_mod  # noqa: E402

calls = []


def fake_detail(platform, source_id, board, posting):
    calls.append(source_id)
    if source_id == "boom":
        raise RuntimeError("tenant refused")
    return ("We are hiring. Applications close on 30 November 2026.",
            None)


# Capture the log rather than letting it print. Two reasons: a passing suite
# that emits {"level": "error"} trains you to ignore real errors, and the
# failure path's logging is worth asserting on rather than merely silencing.
logged = []

real_detail, real_needs = enrich_mod.detail, enrich_mod.needs_detail
real_log, real_log_error = enrich_mod.log, enrich_mod.log_error
enrich_mod.log = lambda event, **f: logged.append((event, f))
enrich_mod.log_error = lambda event, exc, **f: logged.append((event, f))
enrich_mod.detail = fake_detail
enrich_mod.needs_detail = lambda p: p in ("workday", "smartrecruiters")
try:
    ps = [
        Posting(source_id="wd", external_id="1", title="A"),
        # Already has text from its list response: must not cost a request.
        Posting(source_id="gh", external_id="2", title="B",
                description="Apply by 15 January 2027 please."),
    ]
    stats = enrich(ps, sources)
    check("only the platform lacking text is fetched", calls, ["wd"])
    ok("description filled in", ps[0].description is not None)
    check("deadline found in fetched text", ps[0].deadline, "2026-11-30")
    # The rule-based pass covers postings whose text arrived in bulk, too.
    check("deadline found in existing text", ps[1].deadline, "2027-01-15")
    check("both counted", stats["deadlines"], 2)
    check("one fetched", stats["fetched"], 1)

    # Detail endpoints return the WHOLE advert (measured: 4,700-6,400 chars,
    # against ~900 from the adapters that supply text in bulk). Six of those in
    # one batch is ~34,000 characters, which buries the postings in each other.
    # enrich() must excerpt what it fetches.
    #
    # And the deadline must be read BEFORE that excerpting: closing dates sit at
    # the end of an advert, which is precisely what an excerpt discards. Putting
    # the date last here is what makes the ordering testable rather than assumed.
    long_advert = ("We are hiring interns. " * 300
                   + "Applications close on 30 November 2026.")
    calls.clear()

    def long_detail(platform, source_id, board, posting):
        return long_advert, None

    enrich_mod.detail = long_detail
    big = [Posting(source_id="wd", external_id="9", title="D")]
    enrich(big, [Source("wd", "workday")])
    ok("raw advert really is oversized", len(long_advert) > 3000)
    ok("stored description is excerpted", len(big[0].description) < 1200)
    check("deadline read from the full text before excerpting",
          big[0].deadline, "2026-11-30")
    # Proves the ordering rather than a lucky excerpt: the date is not in the
    # text that was kept, yet it was still found.
    ok("date is absent from the excerpt", "30 November" not in big[0].description)
    enrich_mod.detail = fake_detail

    # A tenant that refuses (UOB and Citi both 403) costs that posting its
    # description, never the run its results.
    calls.clear()
    boom = [Posting(source_id="boom", external_id="3", title="C")]
    stats = enrich(boom, [Source("boom", "workday")])
    check("failure counted", stats["failed"], 1)
    check("failure leaves no description", boom[0].description, None)
    check("failure is not fatal", stats["fetched"], 0)

    # A refusal must be visible in the log, not merely survived. A board that
    # silently stops returning descriptions would otherwise look identical to
    # one that has no deadlines to find.
    events = [e for e, _ in logged]
    ok("the refusal was logged", "enrich.failed" in events)
    ok("the posting is named in the log",
       any(f.get("external_id") == "3" for e, f in logged if e == "enrich.failed"))
    ok("every enrich reports a summary", events.count("enrich.done") >= 3)
finally:
    enrich_mod.detail, enrich_mod.needs_detail = real_detail, real_needs
    enrich_mod.log, enrich_mod.log_error = real_log, real_log_error


if failures:
    print(f"{len(failures)} FAILURES:")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("deadline tests passed")
