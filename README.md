# Internship Monitor

Polls ~98 company job boards every six hours and sends a Telegram message when a
posting genuinely matches a stated profile — not when it merely contains
matching keywords.

Runs at zero cost: GitHub Actions for scheduling, free models via OpenRouter for
judgement, and SQLite persisted to an orphan git branch in place of a managed
database.

**The problem it exists for:** about **2% of what these boards publish is an
internship**. The rest is experienced hiring, and no keyword search separates the
remainder reliably — the roles are titled *Summer Analyst*, *Off-Cycle* and
*Industrial Placement* at least as often as *Intern*.

### What arrives

```
Software Engineer, Intern
stripe · Singapore · internship

🟢 Strong fit
✓ Python programming, Building systems at scale, Test automation
✗ Distributed systems experience, Production engineering at large scale

Stripe software engineer internship in Singapore aligns with candidate's
software engineering focus, location, and summer 2027 availability.

⏰ closes in 9 days — 14 Sep 2026

Open posting →
```

The fit band and the skill lines appear only when a resume is attached. Where
the advert states work-authorisation terms, they are quoted verbatim as a
further line.

---

## How it works

```
  98 job boards                    7 ATS adapters, one JSON API each
        │
        ▼
  ~40,000 postings                 fetched in parallel, ~6 minutes
        │
        ▼
  regex prefilter                  deterministic, free, discards ~99%
        │
        ▼
  ~600 candidates                  only now is advert text fetched
        │
        ▼
  LLM classification               batched 3 at a time, fallback chain
        │
        ▼
  Telegram                         new matches only, exactly once
```

The shape of the pipeline is the cost control. A model never sees a posting the
prefilter can rule out for free, and no advert text is downloaded for a posting
no model will read — which is the difference between a handful of API calls per
run and several thousand.

---

## Quick start

```bash
pip install -r requirements.txt
cp .env-template .env          # then fill in OPENROUTER_API_KEY
```

**1. Build your sources and profile**

```bash
python tools/discover_boards.py        # probe companies -> config/sources.yaml
python -m src.intake                   # a short conversation -> config/profile.yaml
python tools/import_resume.py cv.pdf   # optional: adds fit scoring
```

**2. Record a baseline, then monitor**

```bash
python -m src.run --seed    # record what is currently open, notify nothing
python -m src.run           # thereafter, notify on newly appeared postings only
```

`--seed` matters: without it, the first run notifies on every posting currently
open across every board.

**3. Work through the existing backlog**

```bash
python tools/shortlist.py --fresh
```

Writes `shortlist.md`, `.json` and `.csv`. The CSV carries a `status` column and
merges on URL, so re-running preserves what you have already applied to. Roles
closing within a fortnight sort first, then by fit.

### Configuration

`.env`:

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | yes | Model access |
| `TELEGRAM_BOT_TOKEN` | no | Omit for console output instead |
| `TELEGRAM_CHAT_ID` | no | Required alongside the bot token |
| `MODEL` | no | Pins a model to the front of the fallback chain |

`config/profile.yaml` holds what you are looking for — employment type,
locations, fields, availability window, and free-text notes. `python -m
src.intake` writes it for you; it is plain YAML and safe to edit by hand. The
copy in this repository is an example, so the pipeline runs before you have
written your own.

### Commands

| Command | Does |
| --- | --- |
| `python -m src.run` | One monitoring cycle |
| `python -m src.run --dry-run` | Classify and print; persist nothing |
| `python -m src.intake` | Build `profile.yaml` from a description |
| `python tools/import_resume.py` | Parse a CV into the profile, once |
| `python tools/shortlist.py --fresh` | Classify everything open, write md/json/csv |
| `python tools/discover_boards.py` | Find which companies have a public board |
| `python tools/detect_ats.py <url>` | Identify the ATS behind a careers page |
| `python openrouter_free_models.py` | Confirm a `MODEL` override is still free |
| `python tests/run_all.py` | Full offline test suite |
| `python evals/run_eval.py` | Score the classifier against labelled cases |

---

## Design decisions

**Two-stage filtering, for cost.** A deterministic regex prefilter discards ~99%
of the corpus for free; only the remainder reaches a model. The prefilter is
deliberately permissive — people accept roles more broadly than they specify, and
over-fetching is cheaper than never surfacing a posting at all.

**Advert text is fetched only for postings that will be read.** Workday and
SmartRecruiters return no description in their list responses, so without a
second request a third of candidates would be judged on a job title alone.
Enrichment runs *after* the prefilter, making it ~570 extra requests rather than
40,000.

**A rejection has to be repeated before it is believed.** Models are not
deterministic even at temperature 0 — re-judging one posting against a
byte-identical prompt was measured varying ~15 points, and one went
no / yes / yes across three consecutive calls. A single cached rejection would
discard a suitable role on a coin toss. Matches settle on the first verdict,
because the errors are asymmetric: a false match costs one message you delete, a
false rejection costs the job.

**Dates are never taken from the model.** A wrong visa summary is caught the
moment you read the advert; a fabricated deadline is trusted and acted on.
Closing dates come from a structured ATS field where one exists and a regex
otherwise, tested primarily on what it must *ignore* — adverts are full of start
dates, founding years and programme dates.

**Verdicts are matched by an id the model echoes back,** never by position in the
array. Send ten postings, get eight results, and positional matching attaches
verdicts to the wrong roles silently.

**Classification runs against a wall-clock budget.** Free models vary enormously
in how fast they fail; one was measured at 103–125s per attempt while never once
returning valid output. The budget is passed down into each individual call
rather than checked between them, because socket-level timeouts do not bound a
response that trickles in slowly. Stopping early costs a delay and nothing else,
which is what the rule below guarantees.

**A posting is owed a verdict until it has one.** A posting stops being new the
moment it is stored, which is long before anything has judged it, so being new
is no basis for deciding what to classify. Instead, every posting queued for
classification is marked as owed a verdict before the first model call, and the
mark is cleared only by the write that records its verdict. Whatever a run does
not get to — because the budget ran out, because the models were down, because
the job was killed — is picked up by the next one, for as long as the posting is
still listed. Verdicts are written batch by batch as they arrive rather than
once at the end, so a run that stops early keeps everything it has paid for.

**A run that judges nothing is a failed run.** Boards can be healthy while the
model chain is not: free tiers get withdrawn, keys expire, providers refuse. A
run that sent postings for classification and got no verdict back exits
non-zero, which fires the workflow's own alert, and says so on Telegram. Losing
some verdicts is ordinary and stays quiet — they are owed, so they come back.
The daily heartbeat carries the count still waiting, so a backlog that is not
draining is visible before anyone goes looking for it.

**Delivery is exactly-once.** Notification rows are written only after a
confirmed send, stored apart from the classification cache, and never pruned.
Because the row is the record, its absence is the queue: a match with no
notification row is re-offered on the next run until a send is confirmed.

**Resume data is minimised before it leaves the machine.** Contact details,
phone numbers and identity numbers are stripped by rule first; the PDF is then
reduced to a short structured summary by a single model call and never sent
again. The document itself is never stored or committed.

---

## Coverage

Seven adapters — **Greenhouse, Lever, Ashby, SmartRecruiters, Workday, Oracle
Cloud** and **amazon.jobs** — chosen because each publishes a JSON API intended
for consumption. Outside that set, employers are either HTML-only and
bot-protected or closed to automated access outright.

Each adapter absorbs a different quirk of its platform: page sizes that are
silently capped, tenants that serve page 1 indefinitely instead of an empty
result, display fields that look like unique identifiers but are not, and
location strings that range from `SG` to `Central Region (City Area)`.
Normalisation happens once, at the boundary, so every downstream stage deals in
one vocabulary.

An empty result means "nothing on the boards currently tracked", never "nothing
exists" — board identifiers are frequently not company names (Optiver publishes
under `optiverus`, DRW under `drweng`), and some employers run their campus
pipeline on an entirely separate system from their main careers site.

---

## Quality

```bash
python -m ruff check .     # lint; configuration in ruff.toml
python tests/run_all.py    # 543 assertions, offline
```

Both run in CI on every push and in a pre-push hook, so neither a lint failure
nor a red suite reaches the remote.

The tests are offline by construction — no network, no model, no credentials —
which is what makes them fast enough to gate every push. There is no test
framework: the suites are plain scripts that collect failures and exit non-zero,
trading parametrisation and assertion introspection for a dependency-free run
that behaves identically on a laptop and in CI.

| Suite | Covers |
| --- | --- |
| `test_pipeline` | Canonicalisation, employment type, prefilter, dedupe, idempotency, circuit breaker |
| `test_llm` | Fallback chain, retry policy per status class, JSON extraction from prose, backoff, responses that are not completions |
| `test_classify` | Verdict matching by echoed id, batch-poisoning recovery, injection escaping |
| `test_fetchers` | Seven adapters replayed against recorded responses; detail endpoints |
| `test_notify` | Message rendering, HTML escaping, notifier selection, heartbeat thresholds |
| `test_resume` | PDF extraction and its failure modes, redaction, prompt-size caps |
| `test_deadline` | Deadline extraction and the dates it must ignore; enrichment isolation |
| `test_reclassify` | Rejection confirmation, judgement counting, retry bounds |
| `test_run` | Empty-board detection, delivery outcomes, redelivery bounds |

Fixtures in `tests/fixtures/` are trimmed real API responses. They catch a
refactor silently dropping a field; they cannot catch a vendor changing its
schema.

### Evals

Tests check that the code works. They cannot check whether the classifier's
*judgement* is any good — that needs labelled examples and a real model.

```bash
python evals/run_eval.py                      # all 16 cases
python evals/run_eval.py --tag keyword-trap   # a targeted subset
```

`evals/cases.yaml` holds 16 real postings and the correct answer for each. The
harness feeds them to the live classifier with a real profile and reports a pass
rate, then exits non-zero below a threshold. Worth knowing:

- **Each case runs three times.** The model gives different answers to the same
  question, so a single run proves nothing. A case passing 2 of 3 says something
  real about stability.
- **Score bounds are deliberately loose.** Scores wander ~15 points on their
  own; the bounds are there to catch a collapse, not drift.
- **Eval cases never appear in the prompt as examples.** They would pass for free
  and measure nothing. The worked examples in `classify.py` are written
  separately for exactly this reason.

Evals live outside `tests/` because they need an API key, need the network, and
are not repeatable — everything the pre-push hook must not be.

---

## Deployment

`monitor.yml` runs every six hours on GitHub Actions. `state.db` is restored from
an orphan `state` branch at the start of each run and force-pushed back at the
end, so history holds exactly one copy of the database rather than a new one per
run. The save step runs even when the job fails, because a run may have sent
notifications it has not yet recorded.

Failures are reported by the workflow itself rather than from inside the
application — a `curl` step conditioned on `failure()`, depending on no
interpreter and no database, so nothing it needs can be killed alongside the job.
A daily heartbeat covers the opposite case: absence of alerts should never be
indistinguishable from absence of the system.

`keepalive.yml` records activity monthly, because GitHub disables scheduled
workflows after 60 days of repository inactivity.

**Run your own copy privately.** Once `profile.yaml` describes you and
`state.db` holds what you have been matched with, the repository contains
personal data: what you are looking for, what you told it about your right to
work, and every role it has ever sent you. This public copy carries an example
profile and no database for exactly that reason.

---

## Project layout

```
src/
  run.py              orchestration only, no business logic
  config.py           profile.yaml / sources.yaml loading, profile_hash
  models.py           every shape that crosses a boundary
  fetchers/           one adapter per ATS, behind a registry
  normalise.py        canonical locations, employment type, excerpts, deadlines
  prefilter.py        profile -> deterministic keep/drop rules
  enrich.py           advert text + closing dates, for prefilter survivors only
  classify.py         the prompt, batching, PROMPT_VERSION
  llm.py              fallback chain, JSON extraction, tracing
  store.py            all SQL
  notify.py           message rendering, Telegram HTML with a plain fallback
  observability.py    structured logs, heartbeat, degraded-source alert
  intake/             free text -> profile.yaml; PDF -> redact -> summary
tools/                board discovery, ATS detection, resume import, shortlist
tests/                offline: no network, no model, no credentials
evals/                labelled cases judged by the real classifier
.github/workflows/    monitor (6h), tests (on push), keepalive (monthly)
```

~3,700 lines in `src/`, ~1,200 across `tools/` and `evals/`, and ~2,350 of
tests.

**Stack:** Python 3.11, `requests`, `pydantic`, `PyYAML`, `pypdf`, `tqdm`,
`ruff`. SQLite from the standard library. No application framework, no ORM, no
task queue.
