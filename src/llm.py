"""OpenRouter client: model fallback chain, schema validation, tracing.

Every LLM call in the system — intake and monitor alike — goes through call().
Each is stage-tagged so intake failures and classification failures never have
to be grepped apart.
"""

import json
import os
import random
import re
import sys
import time
import uuid
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TypeVar

import requests
from pydantic import BaseModel, ValidationError

from src.config import load_env
from src.models import CallMeta

T = TypeVar("T", bound=BaseModel)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
# (connect, read). Split explicitly: a hung TCP connect should fail in seconds,
# while a free model that is genuinely thinking may legitimately take a minute.
# A single scalar would force one of those two to be wrong.
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 90
TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)

APP_URL = "https://github.com/local/job-hunt"
APP_TITLE = "job-hunt internship monitor"

# OpenRouter stores two distinct identifiers, and they are not interchangeable:
#   session_id    — set via the body param of the same name (or an X-Session-Id
#                   header). This is the one the dashboard groups calls by.
#   external_user — what the `user` body param maps to. Documented for abuse
#                   detection, and NOT usable for grouping.
# Both are sent: session_id for tracking, user so calls are attributable.
# Verified empirically — X-OpenRouter-Session-Id is silently ignored.
SESSION_ID = (os.environ.get("OPENROUTER_SESSION_ID")
              or f"job-hunt-{uuid.uuid4().hex[:12]}")

BACKOFF_CAP = 30


def _backoff(failures: int, retry_after: float | None = None) -> float:
    """Exponential backoff with jitter, counted across the WHOLE call.

    The counter deliberately does not reset when the chain moves to the next
    model: if three models in a row are struggling, the pressure is upstream and
    hammering the next one immediately helps nobody.
    """
    # Only a POSITIVE Retry-After replaces the exponential. A header of 0, or a
    # date already in the past (or one a few seconds of clock skew turns into
    # the past), reads as "wait no time at all" — which would fire all six
    # attempts back to back at a provider that has just rate-limited us, with
    # the failure count growing but never used.
    if retry_after:
        return min(retry_after, BACKOFF_CAP)
    return min(BACKOFF_CAP, (2 ** failures) + random.uniform(0, 0.75))


def _retry_after(value: str | None) -> float | None:
    """Seconds to wait from a Retry-After header, or None if it cannot be read.

    The header is either a number of seconds or an HTTP date (RFC 9110). Only
    the first form used to be handled: float() on a date raised ValueError,
    which nothing between here and main() catches, so a single 429 carrying a
    date would have ended the run and discarded every verdict it had made.
    """
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())

# Free models, ordered. Small free models fail by returning malformed JSON far
# more often than they time out, so the chain is the retry strategy: move to the
# next model rather than trying to regex a valid object out of bad output.
# minimax/minimax-m3:free was the workhorse (651 successful calls) until its free
# tier was withdrawn: every request now 404s with "This model is unavailable for
# free". Removed rather than left in place — a dead first choice costs a wasted
# call on every single classification, and 4xx is never retried, so the failure
# is silent apart from the trace.
#
# The chain is verified against the real classification prompt, not chosen from
# the model list. Most free models fail this task for reasons a spec sheet does
# not show: several 429 upstream continuously, one 400s, two return empty
# content, and one is restricted to "agentic harnesses" and 403s a plain API
# call.
# nemotron-3.5-lightning was also removed, and for a more expensive reason than
# minimax: it does not fail fast. It is a reasoning model that narrates before
# answering ("Here's a thinking process: 1. **Analyze User Input:**") and spends
# its whole token budget doing so, so the response is truncated before any JSON
# exists. Measured at 103-125s per attempt, twice per model, never once
# succeeding — four minutes of the run's budget burned per batch to produce
# nothing. A model that fails in 1.3s costs less than one that fails in 125s.
MODEL_CHAIN = {
    "intake": ["nvidia/nemotron-3-super-120b-a12b:free",
               "google/gemma-4-31b-it:free"],
    "classify": ["nvidia/nemotron-3-super-120b-a12b:free",
                 "google/gemma-4-31b-it:free",
                 "google/gemma-4-26b-a4b-it:free"],
}

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class AllModelsFailed(RuntimeError):
    pass


class MalformedResponse(ValueError):
    """An HTTP 200 whose body is not a chat completion."""


def _first_choice(body) -> dict:
    """The first choice of a completion, or MalformedResponse.

    A 200 promises nothing about shape. Indexing straight in turned an empty
    `choices` list into IndexError and a null body or message into TypeError —
    neither of which the retry loop caught, so instead of moving to the next
    model the exception escaped classify() and ended the run.
    """
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        raise MalformedResponse(f"no choices in response: {str(body)[:300]}")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise MalformedResponse(f"choice has no message: {str(choice)[:300]}")
    return choice


def _extract_json(content: str) -> str:
    """Pull the JSON object out of whatever the model actually returned.

    Two things have to be tolerated, both observed in production:
      - a markdown fence, even with response_format set
      - a reasoning model narrating first. Nemotron opens with "Here's a
        thinking process: 1. **Analyze User Input:** ..." and only then emits
        the object, which failed 22 of 32 calls until this looked past it.

    Rejecting those responses threw away answers the model had in fact given.
    """
    content = content.strip()
    m = _FENCE_RE.match(content)
    if m:
        content = m.group(1).strip()
    # Always extract, even when the text already starts with "{": models append
    # commentary after the object as readily as before it, and trailing text is
    # a parse error just the same.
    return _first_object(content) or content


def _first_object(text: str) -> str | None:
    """The first balanced {...}, ignoring braces inside strings.

    Brace counting rather than a regex: prose before the object frequently
    contains braces of its own, and the schema examples the model is echoing
    back are themselves full of them.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _valid_model_id(value: str) -> bool:
    """OpenRouter ids look like 'vendor/model[:tag]'.

    A malformed MODEL (a duplicated 'MODEL=' prefix is the classic .env typo)
    would otherwise 4xx on every call before the chain fell through to a real
    model — silent, and it costs the rate limit on each run.
    """
    return "/" in value and "=" not in value and " " not in value


def model_chain(stage: str) -> list[str]:
    """Chain for a stage; MODEL env var, if set and well-formed, is tried first."""
    chain = list(MODEL_CHAIN.get(stage, MODEL_CHAIN["classify"]))
    load_env()
    preferred = (os.environ.get("MODEL") or "").strip()
    if preferred and not _valid_model_id(preferred):
        print(f"warning: ignoring malformed MODEL={preferred!r} in .env "
              f"(expected e.g. 'vendor/model:free')", file=sys.stderr)
        preferred = ""
    if preferred:
        chain = [preferred] + [m for m in chain if m != preferred]
    return chain


def call(stage: str, messages: list, schema: type[T], run_id: str = "",
         batch_size: int = 1, conn=None, attempts_per_model: int = 2,
         deadline: float | None = None) -> tuple[T, CallMeta]:
    """Call OpenRouter, validate against `schema`, trace every attempt.

    Raises AllModelsFailed only when every model in the chain is exhausted, or
    when `deadline` (a time.monotonic() value) passes first.

    The deadline exists because requests' timeout is per socket operation, not
    per request: a response that trickles in below the read timeout can run
    indefinitely. One was observed taking 861 seconds, which overran a
    240-second caller budget by a factor of four, because the caller can only
    check the clock between calls. This bounds the overshoot to a single
    attempt rather than a whole chain of them.
    """
    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")

    payload_base = {
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 4000,
        # Stamps every call in this process with one id, so a run can be
        # isolated on the OpenRouter activity dashboard.
        "session_id": SESSION_ID,
        "user": SESSION_ID,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_URL,
        "X-Title": APP_TITLE,
        "X-Session-Id": SESSION_ID,  # redundant with the body param; sent anyway
    }
    input_chars = sum(len(m.get("content", "")) for m in messages)

    def trace(model, provider, attempt, status, http_status, latency_ms,
              parsed_ok, excerpt="", generation_id=""):
        if conn is not None:
            from src import store
            store.record_llm_call(conn, run_id, stage, model, provider, attempt,
                                  status, http_status, latency_ms, batch_size,
                                  parsed_ok, excerpt, SESSION_ID, generation_id)

    failures = 0  # shared across the whole call, so backoff grows monotonically

    def past_deadline() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    def nap(seconds: float) -> None:
        """Back off, but never sleep past the deadline — waiting 30 seconds to
        make one more attempt that cannot be made is pure delay."""
        if deadline is not None:
            seconds = min(seconds, max(0.0, deadline - time.monotonic()))
        if seconds > 0:
            time.sleep(seconds)

    for depth, model in enumerate(model_chain(stage)):
        for attempt in range(attempts_per_model):
            if past_deadline():
                raise AllModelsFailed(
                    f"deadline passed for stage={stage} after {failures} "
                    f"failures (session {SESSION_ID})")
            t0 = time.monotonic()
            provider, raw, generation_id = "", "", ""
            try:
                resp = requests.post(CHAT_URL, headers=headers,
                                     json={"model": model, **payload_base},
                                     timeout=TIMEOUT)
                latency = int((time.monotonic() - t0) * 1000)

                if resp.status_code == 429:
                    failures += 1
                    trace(model, "", attempt, "rate_limited", 429, latency, False,
                          resp.text[:2000])
                    nap(_backoff(failures,
                                 _retry_after(resp.headers.get("retry-after"))))
                    continue
                if resp.status_code >= 500:
                    failures += 1
                    trace(model, "", attempt, "transport_error", resp.status_code,
                          latency, False, resp.text[:2000])
                    nap(_backoff(failures))
                    continue
                if resp.status_code >= 400:
                    # 4xx is our bug (bad model id, bad body) — never retried.
                    trace(model, "", attempt, "client_error", resp.status_code,
                          latency, False, resp.text[:2000])
                    break

                body = resp.json()
                choice = _first_choice(body)
                # `or ""` rather than a .get default: a key present with a null
                # value returns None, which CallMeta rejects as a ValidationError
                # — and that is caught below as a parse error, discarding an
                # answer the model had in fact given.
                provider = body.get("provider") or ""
                generation_id = body.get("id") or ""
                raw = choice["message"].get("content") or ""
                parsed = schema.model_validate_json(_extract_json(raw))

                trace(model, provider, attempt, "ok", 200, latency, True,
                      generation_id=generation_id)
                return parsed, CallMeta(
                    model=model, provider=provider, attempt=attempt,
                    fallback_depth=depth, latency_ms=latency,
                    input_chars=input_chars, output_chars=len(raw),
                    finish_reason=choice.get("finish_reason") or "",
                    session_id=SESSION_ID, generation_id=generation_id,
                )

            except (ValidationError, json.JSONDecodeError, KeyError,
                    MalformedResponse) as e:
                # Malformed output is the dominant failure mode for small free
                # models. Retrying the same model rarely helps, so no sleep —
                # the chain moves on quickly.
                failures += 1
                latency = int((time.monotonic() - t0) * 1000)
                # A malformed body has no content to show, so the trace carries
                # the exception's own description of what arrived instead.
                trace(model, provider, attempt, "parse_error", 200, latency,
                      False, f"{type(e).__name__}: {raw[:1800] or str(e)[:1800]}",
                      generation_id)
            except requests.Timeout:
                failures += 1
                latency = int((time.monotonic() - t0) * 1000)
                trace(model, provider, attempt, "timeout", None, latency, False)
                nap(_backoff(failures))
            except requests.RequestException as e:
                failures += 1
                latency = int((time.monotonic() - t0) * 1000)
                trace(model, provider, attempt, "transport_error", None, latency,
                      False, str(e)[:2000])
                nap(_backoff(failures))

    raise AllModelsFailed(
        f"all models exhausted for stage={stage} (session {SESSION_ID})")
