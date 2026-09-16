"""Tests for the OpenRouter wrapper: fallback chain, retries, error branching.

No network: requests.post is replaced with a scripted fake, and time.sleep is
stubbed so backoff does not actually wait.
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from src import llm  # noqa: E402
from src.models import VerdictBatch  # noqa: E402

failures = []


def _json_checks(check, ok):
    """Extraction has to survive what models actually return, not what the API
    contract promises. Every case here was observed in production."""
    ex = llm._extract_json

    # A reasoning model narrating before it answers. Rejecting these threw away
    # answers the model had in fact given.
    check("prose before the object",
          ex('Here\'s a thinking process:\n\n1. **Analyze**\n\n{"a":1}'), '{"a":1}')
    check("prose after the object",
          ex('{"a":1}\n\nThat is my answer.'), '{"a":1}')

    # Braces inside strings must not close the object early — job titles and
    # echoed schema examples both contain them.
    check("brace inside a string",
          ex('prelude {"reason":"pay is {competitive}","b":2} tail'),
          '{"reason":"pay is {competitive}","b":2}')
    check("escaped quote inside a string",
          ex(r'x {"reason":"they said \"yes\"","b":2} y'),
          r'{"reason":"they said \"yes\"","b":2}')
    check("nested objects", ex('note {"a":{"b":{"c":1}}} end'),
          '{"a":{"b":{"c":1}}}')

    # Truncated output is what actually happens when a model spends its whole
    # budget reasoning: there is no complete object, and inventing one would be
    # worse than failing. The caller retries.
    ok("unterminated object is not salvaged",
       ex('Here\'s a thinking process: {"a":') == 'Here\'s a thinking process: {"a":')
    ok("no object at all is passed through unchanged",
       ex('I cannot answer that.') == 'I cannot answer that.')


def check(name, got, want):
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


def ok(name, cond):
    if not cond:
        failures.append(f"{name}: expected true")


# --- pure helpers ---------------------------------------------------------

check("strips ```json fence",
      llm._extract_json('```json\n{"a": 1}\n```'), '{"a": 1}')
check("strips bare fence", llm._extract_json('```\n{"a": 1}\n```'), '{"a": 1}')
check("leaves plain json alone", llm._extract_json('{"a": 1}'), '{"a": 1}')
check("trims whitespace", llm._extract_json('  {"a": 1}  '), '{"a": 1}')
_json_checks(check, ok)

# A model whose free tier has been withdrawn must not sit at the head of the
# chain: 4xx is never retried, so it costs a wasted call on every request and
# fails silently apart from the trace. minimax was removed for exactly this.
ok("no withdrawn model in the chain",
   not any("minimax" in m for chain in llm.MODEL_CHAIN.values() for m in chain))
ok("every stage has a fallback",
   all(len(chain) >= 2 for chain in llm.MODEL_CHAIN.values()))

# A duplicated "MODEL=" prefix is the classic .env typo; it must be rejected
# rather than sent, or every call 4xxs before the chain falls through.
check("rejects MODEL= prefix", llm._valid_model_id("MODEL=vendor/model:free"), False)
check("rejects missing slash", llm._valid_model_id("modelname"), False)
check("rejects spaces", llm._valid_model_id("vendor/model name"), False)
check("accepts a real id", llm._valid_model_id("minimax/minimax-m3:free"), True)

# Backoff grows and is capped; retry_after wins when the server supplies one.
ok("backoff grows", llm._backoff(1) < llm._backoff(3) < llm._backoff(5))
check("backoff capped", llm._backoff(50) <= llm.BACKOFF_CAP, True)
check("retry_after honoured", llm._backoff(3, retry_after=2.0), 2.0)
check("retry_after capped", llm._backoff(1, retry_after=9999), llm.BACKOFF_CAP)
# A Retry-After of zero — a past date, or a literal 0 — must not mean "retry at
# once". Six attempts back to back at a provider that has just rate-limited us
# is the one thing backoff exists to prevent.
ok("zero retry_after falls back to exponential", llm._backoff(3, retry_after=0.0) > 1)

# Retry-After is seconds OR an HTTP date. float() on the date form raised
# ValueError straight out of call(), ending the run over a rate limit.
check("retry-after seconds", llm._retry_after("7"), 7.0)
check("retry-after absent", llm._retry_after(None), None)
check("retry-after unreadable", llm._retry_after("soon"), None)
check("retry-after negative is not a negative sleep", llm._retry_after("-5"), 0.0)
check("retry-after date in the past waits nothing",
      llm._retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), 0.0)
_future = llm._retry_after(
    (datetime.now(UTC) + timedelta(seconds=20))
    .strftime("%a, %d %b %Y %H:%M:%S GMT"))
ok("retry-after date in the future waits until then",
   _future is not None and 15 <= _future <= 20)


# --- scripted transport ---------------------------------------------------

_WELL_FORMED = object()


class FakeResponse:
    def __init__(self, status=200, content='{"results": []}', provider="prov",
                 headers=None, gen_id="gen-1", body=_WELL_FORMED):
        self.status_code = status
        self.headers = headers or {}
        self._content = content
        self._provider = provider
        self._gen_id = gen_id
        # Replaces the whole parsed body, for responses that are not a chat
        # completion at all. A sentinel rather than None, because a JSON null
        # body is itself one of the cases worth testing.
        self._body = body
        self.text = "error body"

    def json(self):
        if self._body is not _WELL_FORMED:
            return self._body
        return {"provider": self._provider, "id": self._gen_id,
                "choices": [{"message": {"content": self._content},
                             "finish_reason": "stop"}]}


class FakeTransport:
    """Returns a scripted response (or raises) per call, recording models seen."""

    def __init__(self, script):
        self.script = list(script)
        self.models = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.models.append(json["model"])
        # Running dry must not look like success: an implicit good response
        # would make "chain exhausted" pass for the wrong reason if the chain
        # ever grows longer than the script.
        item = self.script.pop(0) if self.script else FakeResponse(content="dry")
        if isinstance(item, Exception):
            raise item
        return item


def run_call(script, stage="classify", **kw):
    """Drive llm.call against a scripted transport.

    load_env is stubbed throughout: it repopulates MODEL and the API key from
    the repository .env via setdefault, so without this the suite asserts
    against a different model chain locally than in CI.
    """
    transport = FakeTransport(script)
    real_post, real_sleep, real_load = llm.requests.post, llm.time.sleep, llm.load_env
    llm.requests.post = transport.post
    llm.time.sleep = lambda *_: None
    llm.load_env = lambda *a, **k: None
    try:
        return transport, llm.call(stage, [{"role": "user", "content": "x"}],
                                   VerdictBatch, **kw)
    finally:
        llm.requests.post = real_post
        llm.time.sleep = real_sleep
        llm.load_env = real_load


def chain_for(stage="classify"):
    """model_chain with load_env stubbed, for the same isolation reason."""
    real_load = llm.load_env
    llm.load_env = lambda *a, **k: None
    try:
        return llm.model_chain(stage)
    finally:
        llm.load_env = real_load


import os  # noqa: E402

os.environ["OPENROUTER_API_KEY"] = "test-key"
os.environ.pop("MODEL", None)

GOOD = '{"results": [{"id": "1", "is_match": true}]}'

# Happy path: first model, first attempt.
transport, (parsed, meta) = run_call([FakeResponse(content=GOOD)])
check("happy path parses", len(parsed.results), 1)
check("happy path no fallback", meta.fallback_depth, 0)
check("happy path attempt 0", meta.attempt, 0)
check("generation id captured", meta.generation_id, "gen-1")
check("session id attached", meta.session_id, llm.SESSION_ID)

# Fenced JSON from a model that ignores response_format still parses.
_, (parsed, _) = run_call([FakeResponse(content=f"```json\n{GOOD}\n```")])
check("fenced response parses", len(parsed.results), 1)

# Malformed output: retried on the same model, then the chain moves on. Parse
# failure is the dominant error mode for small free models, so this is the path
# that matters most.
transport, (parsed, meta) = run_call(
    [FakeResponse(content="not json"), FakeResponse(content="still not json"),
     FakeResponse(content=GOOD)])
check("parse error falls through", meta.fallback_depth, 1)
ok("second model differs", transport.models[0] != transport.models[2])

# 429 is retried; the request is not abandoned.
transport, (parsed, meta) = run_call(
    [FakeResponse(status=429, headers={"retry-after": "1"}),
     FakeResponse(content=GOOD)])
check("429 retried then succeeds", len(parsed.results), 1)
check("429 stayed on first model", transport.models[0], transport.models[1])

# A 429 whose Retry-After is an HTTP date is retried like any other, not raised.
_, (parsed, _) = run_call(
    [FakeResponse(status=429,
                  headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}),
     FakeResponse(content=GOOD)])
check("429 with a date retry-after retried", len(parsed.results), 1)

# 5xx is retried too.
_, (parsed, _) = run_call([FakeResponse(status=503), FakeResponse(content=GOOD)])
check("5xx retried", len(parsed.results), 1)

# A 200 that is not a chat completion is a bad answer, not a crash: the chain
# moves on exactly as it does for unparseable JSON. Each of these used to raise
# IndexError or TypeError out of call(), past classify(), and end the run.
for label, body in [("empty choices", {"id": "gen-x", "choices": []}),
                    ("error object, no choices", {"error": {"code": 502}}),
                    ("null body", None),
                    ("null message", {"choices": [{"message": None}]}),
                    ("choice not an object", {"choices": ["oops"]})]:
    try:
        _, (parsed, meta) = run_call([FakeResponse(body=body),
                                      FakeResponse(body=body),
                                      FakeResponse(content=GOOD)])
        check(f"{label}: falls through to the next model", meta.fallback_depth, 1)
    except Exception as e:
        failures.append(f"{label}: raised {type(e).__name__}: {e}")

raised = None
try:
    run_call([FakeResponse(body={"choices": []})] * 12)
except Exception as e:
    raised = type(e)
check("an all-malformed chain raises AllModelsFailed, nothing else",
      raised, llm.AllModelsFailed)

# Null metadata beside a valid answer must not throw that answer away. CallMeta
# rejects None for its string fields, and the ValidationError that raised was
# caught as a parse error — so a correct response counted as a failure.
_, (parsed, meta) = run_call([FakeResponse(body={
    "id": None, "provider": None,
    "choices": [{"message": {"content": GOOD}, "finish_reason": None}]})])
check("null metadata keeps the answer", meta.fallback_depth, 0)
check("null finish_reason recorded as empty", meta.finish_reason, "")

# 4xx is OUR bug (bad model id, bad body) and must not be retried on the same
# model — it would burn the rate limit on a request that cannot succeed.
transport, (parsed, meta) = run_call(
    [FakeResponse(status=400), FakeResponse(content=GOOD)])
check("4xx not retried on same model", transport.models[0] != transport.models[1], True)
check("4xx moves to next model", meta.fallback_depth, 1)

# Timeouts and connection errors are retried.
_, (parsed, _) = run_call([requests.Timeout(), FakeResponse(content=GOOD)])
check("timeout retried", len(parsed.results), 1)
_, (parsed, _) = run_call([requests.ConnectionError(), FakeResponse(content=GOOD)])
check("connection error retried", len(parsed.results), 1)

# Everything failing raises rather than returning something wrong.
raised = False
try:
    run_call([FakeResponse(content="bad")] * 12)
except llm.AllModelsFailed:
    raised = True
check("exhausted chain raises AllModelsFailed", raised, True)

# A malformed MODEL is ignored with a warning rather than being sent.
DEFAULT_CHAIN = list(llm.MODEL_CHAIN["classify"])
os.environ["MODEL"] = "MODEL=vendor/model:free"
chain = chain_for()
ok("malformed MODEL not in chain", "MODEL=vendor/model:free" not in chain)
check("malformed MODEL leaves chain intact", chain, DEFAULT_CHAIN)

# A well-formed MODEL not already present is prepended.
os.environ["MODEL"] = "vendor/custom:free"
chain = chain_for()
check("valid MODEL leads chain", chain[0], "vendor/custom:free")
check("prepending keeps the rest", chain[1:], DEFAULT_CHAIN)

# A MODEL that IS already in the chain must be moved, not duplicated — this is
# the case that actually exercises the dedup.
os.environ["MODEL"] = DEFAULT_CHAIN[-1]
chain = chain_for()
check("existing MODEL moved to front", chain[0], DEFAULT_CHAIN[-1])
check("existing MODEL not duplicated", chain.count(DEFAULT_CHAIN[-1]), 1)
check("chain length unchanged", len(chain), len(DEFAULT_CHAIN))
os.environ.pop("MODEL", None)

# A missing key is an error, not a silent no-op. load_env is stubbed out too:
# otherwise it repopulates the key straight back from the real .env on disk.
saved = os.environ.pop("OPENROUTER_API_KEY")
raised = False
try:
    run_call([FakeResponse(content=GOOD)])
except RuntimeError:
    raised = True
finally:
    os.environ["OPENROUTER_API_KEY"] = saved
check("missing API key raises", raised, True)


if failures:
    print(f"{len(failures)} FAILURES:")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("llm tests passed")
