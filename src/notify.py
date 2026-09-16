"""Notifier interface. Console now; Telegram slots in behind the same protocol."""

import html
import os
from datetime import date
from typing import Protocol

from src.models import Posting, Verdict
from src.normalise import days_until

TELEGRAM_LIMIT = 4000  # real cap is 4096; leave headroom for formatting


class Notifier(Protocol):
    def send(self, posting: Posting, verdict: Verdict) -> bool:
        """Return True only on confirmed delivery — the caller records the
        notification only after that, so a crash costs a duplicate, not a drop."""
        ...

    def send_text(self, text: str) -> bool:
        """Send a plain operational message (heartbeat, failure alert)."""
        ...


# One table, three renderings. Previously the same four values were spelled out
# separately in the plain message, the HTML message and the shortlist — so a new
# WorkAuthorization value meant three edits, and missing one showed up as a
# silently blank label rather than an error.
#
# "not_mentioned" is deliberately absent: it is the overwhelming default and
# says nothing, so it renders as no line at all rather than as noise on every
# single message.
AUTHORIZATION = {
    "sponsorship_offered":
        ("\U0001F6C2", "visa sponsorship offered", "SPONSORSHIP OFFERED"),
    "citizen_or_pr_required":
        ("\U0001F6AB", "citizens / PR only", "CITIZENS/PR ONLY"),
    "authorization_required":
        ("⚠️", "must already have work authorization", "NEEDS OWN WORK AUTH"),
    "unclear":
        ("❓", "work authorization unclear", "AUTH UNCLEAR"),
}


def authorization_label(value: str) -> str | None:
    entry = AUTHORIZATION.get(value)
    return entry[1] if entry else None


# Bands, not numbers. Re-scoring one posting three times against an identical
# prompt gives a spread of roughly ±15 points: the model is a hosted MoE, and
# greedy decoding still flips on near-ties when server-side batching changes the
# logits. Printing "84%" therefore advertises a precision the measurement does
# not have. Three bands are wide enough to survive that noise, and a band is
# what you act on anyway.
#
# The underlying score is still stored, and still sorts the shortlist. Only the
# display was changed, to claim no more than the number can support.
FIT_BANDS = (
    (70, "\U0001F7E2", "Strong fit", "strong"),
    (50, "\U0001F7E1", "Possible fit", "possible"),
    (0, "\U0001F7E0", "Stretch", "stretch"),
)


def fit_band(score: int) -> tuple[str, str, str]:
    """(icon, label, slug) for a 0-100 score."""
    for floor, icon, label, slug in FIT_BANDS:
        if score >= floor:
            return icon, label, slug
    return FIT_BANDS[-1][1:]


def deadline_line(posting: Posting) -> str | None:
    """'closes in 9 days - 14 Sep 2026', or None when no deadline is known.

    Days remaining is computed at send time rather than stored, so a message
    never reports a countdown that was accurate when the row was written.
    """
    left = days_until(posting.deadline)
    if left is None:
        return None
    try:
        pretty = date.fromisoformat(posting.deadline).strftime("%d %b %Y")
    except (ValueError, TypeError):
        pretty = posting.deadline
    if left < 0:
        return f"closed {pretty}"
    if left == 0:
        return f"closes TODAY - {pretty}"
    return f"closes in {left} day{'s' if left != 1 else ''} - {pretty}"


def format_message(posting: Posting, verdict: Verdict) -> str:
    location = posting.location or posting.location_raw or "location unknown"
    lines = [
        f"{posting.title}",
        f"{posting.source_id} — {location} — "
        f"{posting.employment_type or 'type unknown'}",
    ]
    if verdict.fit_score is not None:
        # Shown as evidence, not as a verdict: a band invites checking the
        # skills beside it against the advert, where a number invites trusting
        # it. Omitted entirely when no resume is configured.
        fit = fit_band(verdict.fit_score)[1]
        if verdict.matched_skills:
            fit += " — has " + ", ".join(verdict.matched_skills[:3])
        if verdict.missing_skills:
            fit += "; lacks " + ", ".join(verdict.missing_skills[:3])
        lines.append(fit)
    closing = deadline_line(posting)
    if closing:
        lines.append(closing)
    if verdict.reason:
        lines.append(verdict.reason)
    label = authorization_label(verdict.work_authorization)
    if label:
        lines.append(f"[{label}]")
    if posting.url:
        lines.append(posting.url)
    return "\n".join(lines)[:TELEGRAM_LIMIT]


# --- Telegram HTML rendering ----------------------------------------------
# Telegram's HTML mode is used rather than MarkdownV2: MarkdownV2 requires
# escaping seventeen characters, several of which (-, ., (, )) appear in almost
# every job title, and one missed escape rejects the entire message.


def company_of(source_id: str) -> str:
    """'imc-greenhouse' -> 'imc'. The ATS name is plumbing, not information."""
    return source_id.rsplit("-", 1)[0]


def format_html(posting: Posting, verdict: Verdict) -> str:
    """The Telegram message. Every field is escaped and length-capped.

    Capping each component rather than truncating the finished string is
    deliberate: cutting the assembled HTML mid-tag produces a malformed entity,
    which Telegram rejects outright — losing the whole notification to a long
    job title.
    """
    esc = html.escape
    location = posting.location or posting.location_raw or "location unknown"
    etype = (posting.employment_type or "type unknown").replace("_", " ")

    lines = [
        f"<b>{esc(posting.title.strip()[:150])}</b>",
        f"{esc(company_of(posting.source_id))} · {esc(location[:60])}"
        f" · {esc(etype)}",
    ]

    if verdict.fit_score is not None:
        icon, label, _ = fit_band(verdict.fit_score)
        lines += ["", f"{icon} <b>{label}</b>"]
        if verdict.matched_skills:
            lines.append("✓ " + esc(", ".join(verdict.matched_skills[:3])[:200]))
        if verdict.missing_skills:
            lines.append("✗ " + esc(", ".join(verdict.missing_skills[:3])[:200]))

    closing = deadline_line(posting)
    if closing:
        left = days_until(posting.deadline)
        # Bold under a fortnight: at that point the deadline outranks the fit
        # band as the thing you need to act on.
        urgent = left is not None and 0 <= left <= 14
        text = f"<b>{esc(closing)}</b>" if urgent else esc(closing)
        lines += ["", f"⏰ {text}"]

    if verdict.reason:
        lines += ["", f"<i>{esc(verdict.reason.strip()[:400])}</i>"]

    entry = AUTHORIZATION.get(verdict.work_authorization)
    if entry:
        icon, label, _ = entry
        lines += ["", f"{icon} {label}"]
        if verdict.authorization_quote:
            lines.append(
                f"<blockquote>{esc(verdict.authorization_quote.strip()[:250])}"
                "</blockquote>")

    if posting.url:
        lines += ["", f'<a href="{esc(posting.url, quote=True)}">'
                      "Open posting →</a>"]

    return "\n".join(lines)


class ConsoleNotifier:
    """Default sink: prints matches. Always 'delivers'."""

    def send(self, posting: Posting, verdict: Verdict) -> bool:
        print("\n" + format_message(posting, verdict), flush=True)
        return True

    def send_text(self, text: str) -> bool:
        print("\n" + text, flush=True)
        return True


class TelegramNotifier:
    """Telegram delivery. Selected by make_notifier() whenever both secrets are
    present, so run.py never names a transport."""

    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        if not self.token or not self.chat_id:
            raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")

    def send(self, posting: Posting, verdict: Verdict) -> bool:
        if self.send_text(format_html(posting, verdict), parse_mode="HTML"):
            return True
        # Telegram rejects a message outright if it cannot parse the entities.
        # Losing a job to a formatting problem is the worst outcome available
        # here, so retry once as unformatted text.
        return self.send_text(format_message(posting, verdict))

    def send_text(self, text: str, parse_mode: str | None = None) -> bool:
        import time

        import requests

        payload = {"chat_id": self.chat_id, "text": text[:TELEGRAM_LIMIT],
                   "disable_web_page_preview": True}
        if parse_mode:
            payload["parse_mode"] = parse_mode

        for attempt in range(3):
            try:
                r = requests.post(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    json=payload, timeout=(10, 20),
                )
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429:
                # Telegram allows ~20 messages/min per chat and tells you how
                # long to wait; ignoring retry_after just earns another 429.
                time.sleep(int(r.json().get("parameters", {}).get("retry_after", 5)))
                continue
            if r.status_code == 400 and parse_mode:
                # Bad entities. Retrying identically cannot help; let the caller
                # fall back to plain text rather than burning two more attempts.
                return False
            return r.ok
        return False


def make_notifier() -> Notifier:
    """Telegram when its secrets are present, console otherwise.

    Chosen by environment rather than by a flag so the same command works on a
    laptop and in CI — but note that in CI the console notifier writes to a log
    nobody reads, which is a silent no-op dressed up as success.
    """
    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        return TelegramNotifier()
    return ConsoleNotifier()
