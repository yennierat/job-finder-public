"""Tests for message formatting, notifier selection, heartbeat and alerting.

No network: the Telegram transport is faked. These cover the parts that decide
whether you hear about a job at all, and whether you would notice the system
dying.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import notify, observability, store  # noqa: E402
from src.models import Posting, Verdict  # noqa: E402

failures = []


def check(name, got, want):
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


def ok(name, cond):
    if not cond:
        failures.append(f"{name}: expected true")


def posting(**kw):
    base = dict(source_id="imc-greenhouse", external_id="1",
                title="Software Engineer Intern", location="Singapore",
                employment_type="internship", url="https://example.com/job/1")
    return Posting(**{**base, **kw})


# --- message formatting ---------------------------------------------------

msg = notify.format_message(posting(), Verdict(id="1", is_match=True,
                                               reason="Matches profile."))
ok("includes title", "Software Engineer Intern" in msg)
ok("includes location", "Singapore" in msg)
ok("includes reason", "Matches profile." in msg)
ok("includes url", "https://example.com/job/1" in msg)

# "not_mentioned" is the overwhelming default and says nothing useful, so it is
# omitted rather than added to every single message.
ok("omits not_mentioned", "not_mentioned" not in msg and "[" not in msg)

# The plain message, the HTML message and the shortlist all render work
# authorization from one table. That is only safe if the table is complete: a
# new WorkAuthorization value with no entry renders as a silently missing line,
# not an error. This is the check that turns that into a test failure.
from typing import get_args  # noqa: E402

from src.models import WorkAuthorization  # noqa: E402

# "not_mentioned" is intentionally absent from the table — it renders as no
# line at all — so it is added back before comparing against the enum.
check("every authorization value is renderable",
      sorted(set(notify.AUTHORIZATION) | {"not_mentioned"}),
      sorted(get_args(WorkAuthorization)))
ok("every entry has icon, label and badge",
   all(len(v) == 3 and all(v) for v in notify.AUTHORIZATION.values()))

for authz, expected in [
    ("sponsorship_offered", "visa sponsorship offered"),
    ("citizen_or_pr_required", "citizens / PR only"),
    ("authorization_required", "must already have work authorization"),
    ("unclear", "work authorization unclear"),
]:
    m = notify.format_message(posting(), Verdict(id="1", is_match=True,
                                                 work_authorization=authz))
    ok(f"shows {authz}", expected in m)

# Telegram silently 400s past 4096 characters, so messages must be capped.
long_msg = notify.format_message(
    posting(title="X" * 6000), Verdict(id="1", is_match=True))
ok("truncated below telegram limit", len(long_msg) <= notify.TELEGRAM_LIMIT)

# A posting with no canonical location falls back rather than printing None.
m = notify.format_message(posting(location=None, location_raw="Planet Zorg"),
                          Verdict(id="1", is_match=True))
ok("falls back to raw location", "Planet Zorg" in m)
m = notify.format_message(posting(location=None, location_raw=None),
                          Verdict(id="1", is_match=True))
ok("handles no location at all", "None" not in m)


# --- telegram html rendering ----------------------------------------------

h = notify.format_html(posting(), Verdict(
    id="1", is_match=True, reason="Matches profile.", fit_score=84,
    matched_skills=["PyTorch", "Docker"], missing_skills=["Rust"],
    work_authorization="sponsorship_offered",
    authorization_quote="We sponsor visas."))
ok("html bolds the title", "<b>Software Engineer Intern</b>" in h)
ok("html shows the band", "<b>Strong fit</b>" in h)
# The raw number is deliberately absent: re-scoring the same posting varies by
# roughly +/-15 points, so printing "84%" claims precision that is not there.
ok("html hides the raw number", "84" not in h)
ok("html lists matched skills", "PyTorch" in h)
ok("html lists gaps", "Rust" in h)
ok("html links the posting", '<a href="https://example.com/job/1">' in h)
ok("html quotes the authz clause", "<blockquote>" in h)
# The ATS name is plumbing; 'imc-greenhouse' must read as 'imc'.
check("company stripped of ats", notify.company_of("imc-greenhouse"), "imc")
check("company keeps hyphenated names",
      notify.company_of("jane-street-lever"), "jane-street")

# Telegram rejects the WHOLE message on a malformed entity, so every field that
# reaches it must be escaped. A title containing & or < is not exotic: "R&D
# Intern" is real, and losing that job to an ampersand would be silent.
eh = notify.format_html(
    posting(title="R&D <Intern> \"AI\"", location="Zürich"),
    # fit_score is required for the skill lines to render at all — they are
    # evidence for the score, so they never appear without one.
    Verdict(id="1", is_match=True, reason="a & b <c>", fit_score=70,
            matched_skills=["C++ & Rust"], missing_skills=["<none>"]))
ok("title escaped", "&amp;" in eh and "&lt;Intern&gt;" in eh)
ok("no raw ampersand survives", " & " not in eh)
ok("reason escaped", "a &amp; b" in eh)
ok("skills escaped", "C++ &amp; Rust" in eh)
ok("unicode location preserved", "Zürich" in eh)

# A url with a quote in it would otherwise break out of the href attribute.
qh = notify.format_html(posting(url='https://x.test/a"onmouseover="x'),
                        Verdict(id="1", is_match=True))
ok("url attribute escaped", '"onmouseover=' not in qh)

# Every component is capped, so the assembled message cannot need truncating —
# cutting HTML mid-tag is what produces the malformed entity in the first place.
big = notify.format_html(
    posting(title="T" * 900, location="L" * 400),
    Verdict(id="1", is_match=True, reason="R" * 3000, fit_score=50,
            matched_skills=["m" * 500] * 5, missing_skills=["g" * 500] * 5,
            work_authorization="unclear", authorization_quote="q" * 900))
ok("html message fits the limit", len(big) <= notify.TELEGRAM_LIMIT)
ok("html tags are balanced", big.count("<b>") == big.count("</b>"))

# No resume configured: no band, no empty fit line.
nofit = notify.format_html(posting(), Verdict(id="1", is_match=True,
                                              reason="Matches."))
ok("no band without a score", "fit" not in nofit.lower())

# Bands must be wide enough to absorb the model's own variance, and the
# boundaries must be exact — a posting scoring 70 twice must not land in two
# different bands because of an off-by-one.
check("70 is the strong floor", notify.fit_band(70)[2], "strong")
check("69 is possible", notify.fit_band(69)[2], "possible")
check("50 is the possible floor", notify.fit_band(50)[2], "possible")
check("49 is a stretch", notify.fit_band(49)[2], "stretch")
check("0 is a stretch", notify.fit_band(0)[2], "stretch")
check("100 is strong", notify.fit_band(100)[2], "strong")
ok("each band has its own icon",
   len({notify.fit_band(s)[0] for s in (90, 60, 30)}) == 3)


# --- notifier selection ---------------------------------------------------

saved = {k: os.environ.pop(k, None)
         for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")}
check("console without secrets", type(notify.make_notifier()).__name__,
      "ConsoleNotifier")

os.environ["TELEGRAM_BOT_TOKEN"] = "t"
check("console with only a token", type(notify.make_notifier()).__name__,
      "ConsoleNotifier")

os.environ["TELEGRAM_CHAT_ID"] = "c"
check("telegram with both", type(notify.make_notifier()).__name__,
      "TelegramNotifier")

# Both notifiers satisfy the same protocol; run.py calls send_text on either.
ok("console has send_text", hasattr(notify.ConsoleNotifier(), "send_text"))
ok("telegram has send_text", hasattr(notify.make_notifier(), "send_text"))

for k, v in saved.items():
    if v is None:
        os.environ.pop(k, None)
    else:
        os.environ[k] = v


# --- heartbeat and degraded alerts ---------------------------------------

class Fake:
    def __init__(self):
        self.msgs = []

    def send_text(self, text):
        self.msgs.append(text)
        return True

    def send(self, posting, verdict):
        return True


tmpdir = tempfile.TemporaryDirectory()
conn = store.connect(Path(tmpdir.name) / "t.db")
n = Fake()

first = observability.send_ops_messages(conn, n, 93, 0)
check("heartbeat on first run", first["heartbeat"], True)
check("no alert when healthy", first["alert"], False)

# Four runs a day must not produce four heartbeats.
second = observability.send_ops_messages(conn, n, 93, 0)
check("heartbeat not repeated same day", second["heartbeat"], False)

# More than half the sources failing is systemic and worth interrupting for.
degraded = observability.send_ops_messages(conn, n, 33, 60)
check("alert when majority fail", degraded["alert"], True)
ok("alert names the numbers", "60" in n.msgs[-1] and "93" in n.msgs[-1])

# A week-long outage must not produce 28 identical alerts.
again = observability.send_ops_messages(conn, n, 33, 60)
check("alert rate limited", again["alert"], False)

# Just under half is not systemic.
store.set_meta(conn, observability.ALERT_KEY, "")
mild = observability.send_ops_messages(conn, n, 50, 48)
check("no alert below the threshold", mild["alert"], False)

# A naive timestamp (SQLite's own datetime()) must not crash the run.
conn.execute("UPDATE meta SET value=datetime('now','-2 days') WHERE key=?",
             (observability.HEARTBEAT_KEY,))
conn.commit()
next_day = observability.send_ops_messages(conn, n, 93, 0)
check("heartbeat returns next day", next_day["heartbeat"], True)

# A run where every model failed must not pass for healthy. Sources stay green,
# the run finishes, and "0 matches today" is what a quiet week looks like — the
# 7 Sep outage was invisible for 36 hours for exactly this reason.
dead = observability.send_ops_messages(conn, n, 93, 0, verdicts_requested=12,
                                       verdicts_returned=0)
check("no verdicts at all is an alert", dead["alert"], True)
ok("the alert says how many went unjudged", "12" in n.msgs[-1])

# Its own cooldown, separate from the source alert: two unrelated failures
# sharing one stamp means whichever fires first silences the other for a day.
check("the classification alert is rate limited too",
      observability.send_ops_messages(conn, n, 93, 0, verdicts_requested=12,
                                      verdicts_returned=0)["alert"], False)
store.set_meta(conn, observability.ALERT_KEY, "")
both = observability.send_ops_messages(conn, n, 33, 60, verdicts_requested=12,
                                       verdicts_returned=0)
check("a source outage still alerts while classification is cooling down",
      both["alert"], True)
ok("and names the sources, not the models",
   "60" in n.msgs[-1] and "unjudged" not in n.msgs[-1])

# Losing some is ordinary on free models, and they are retried next run.
store.set_meta(conn, observability.CLASSIFY_ALERT_KEY, "")
partial = observability.send_ops_messages(conn, n, 93, 0, verdicts_requested=12,
                                          verdicts_returned=5)
check("a partial return is not an alert", partial["alert"], False)

# One posting that no model can stomach is not an outage. classify.py already
# retries a poisoned batch item by item, and a quiet run has only one or two
# postings to judge — alerting there means a false alarm every few days.
check("a single unjudged posting is not systemic",
      observability.send_ops_messages(conn, n, 93, 0, verdicts_requested=1,
                                      verdicts_returned=0)["alert"], False)
check("nor are two", observability.exit_code(93, 0, 2, 0), 0)
check("a full batch unjudged is", observability.exit_code(93, 0, 3, 0), 1)

report = observability.health_report(conn)
ok("report mentions sources", "sources" in report)
ok("report mentions matches", "matches" in report)
# The heartbeat has to carry the classification half too, or "still alive" says
# nothing about whether anything is being judged.
ok("report mentions the unjudged backlog", "awaiting a verdict" in report)

# Exit code is non-zero only on systemic failure, so one dead board stays quiet.
check("healthy run exits 0", observability.exit_code(93, 0), 0)
check("one failure still exits 0", observability.exit_code(92, 1), 0)
check("majority failure exits 1", observability.exit_code(40, 58), 1)
check("no sources at all exits 1", observability.exit_code(0, 0), 1)
check("judging nothing of what was requested exits 1",
      observability.exit_code(93, 0, 12, 0), 1)
check("judging some of it does not", observability.exit_code(93, 0, 12, 5), 0)
check("a run with nothing to judge is not a failure",
      observability.exit_code(93, 0, 0, 0), 0)
conn.close()
tmpdir.cleanup()


if failures:
    print(f"{len(failures)} FAILURES:")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("notify tests passed")
