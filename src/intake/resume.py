"""PDF resume -> structured Resume, parsed exactly once.

Three stages, deliberately separate so each can be tested and each fails for a
distinguishable reason:

    extract_text()  PDF bytes -> plain text          (pypdf, no network)
    redact()        plain text -> plain text         (no network, no LLM)
    parse()         plain text -> Resume             (one LLM call, ever)

"Exactly once" is the whole design. A resume is ~4,000 characters; sending it
with every classification batch would multiply token use by the number of
batches and crowd the job adverts out of a small free model's context. So it is
reduced once, at intake, to a summary of a few hundred characters that is cheap
to carry into every prompt thereafter.
"""

import re
from datetime import date
from pathlib import Path

from src.llm import call
from src.models import Resume

# A two-page CV is ~4k characters. The cap is generous enough for a five-page
# academic CV and mean enough to stop a 60-page portfolio PDF from being sent to
# a free model that will refuse it.
MAX_CHARS = 20_000
# Below this, the PDF is almost certainly scanned images rather than text. Better
# to say so than to send 40 characters of noise to the model and store whatever
# it hallucinates from them.
MIN_CHARS = 200

SYSTEM = """You convert a resume into a structured summary used to judge whether job postings suit this person.

Return ONLY JSON:
{
  "headline": "<=150 chars: who they are now, e.g. 'Penultimate-year CS undergraduate, backend and ML'",
  "skills": ["capabilities, e.g. 'distributed systems', 'model fine-tuning'"],
  "tools": ["concrete technologies, e.g. 'Python', 'PyTorch', 'Kubernetes'"],
  "experience": [{"role":"", "organisation":"", "period":"", "highlights":["what they actually built or achieved"]}],
  "projects": ["one line each, only substantial ones"],
  "education": ["degree, institution, expected completion"],
  "years_experience": <number or null: internships and part-time work included>
}

Rules:
- Extract only what the document states. Do not infer a skill from a job title,
  and do not add technologies that would plausibly accompany the ones listed.
  A fabricated skill produces a fit score that is confidently wrong.
- Keep every list to at most 12 entries and every string short. This summary is
  carried into other prompts, so length is a real cost.
- Prefer specific over generic: "PyTorch" over "machine learning frameworks",
  and omit "teamwork", "communication" and similar filler entirely.
- Omit all contact details, addresses and identity numbers even if present.
"""

# Applied before the text ever leaves the machine. The resume goes to a
# third-party API, and a phone number contributes nothing to whether a job fits.
_REDACTIONS = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    # +65 9123 4567, (415) 555-0132, 07700 900123 — seven or more digits with
    # the usual separators. Deliberately not matching shorter runs: "2027" and
    # "GPA 4.85" must survive.
    (re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)"), "[phone]"),
    # Singapore NRIC/FIN, and lookalike national ids.
    (re.compile(r"(?<!\w)[STFGM]\d{7}[A-Z](?!\w)"), "[id]"),
)


class ResumeError(RuntimeError):
    """Raised for anything the user can fix by supplying a different file."""


def extract_text(path: Path) -> str:
    """Read a PDF into plain text.

    pypdf is a pure-Python dependency on purpose: this must install on a laptop
    and in CI without system packages, and the alternative (poppler, tesseract)
    is a toolchain to support forever for one command run once.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - depends on install state
        raise ResumeError(
            "pypdf is not installed — run: pip install -r requirements.txt") from e

    path = Path(path)
    if not path.exists():
        raise ResumeError(f"no such file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ResumeError(f"expected a .pdf, got {path.suffix or 'no extension'}")

    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as e:
        raise ResumeError(f"could not read {path.name}: {type(e).__name__}: {e}") from e

    text = clean("\n".join(pages))
    if len(text) < MIN_CHARS:
        raise ResumeError(
            f"{path.name} yielded only {len(text)} characters of text. It is "
            "probably a scan or an image export rather than a text PDF — "
            "re-export it from the original document, or 'Print to PDF' it.")
    return text[:MAX_CHARS]


def clean(text: str) -> str:
    """Collapse the whitespace PDF extraction produces.

    Two-column CV layouts extract as ragged lines with runs of spaces; left as
    is, that is a large fraction of the tokens sent and none of the meaning.
    """
    text = text.replace("\xa0", " ").replace("•", "- ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def redact(text: str) -> str:
    """Strip contact details before the text leaves this machine."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def parse(text: str, source_file: str = "", today: date | None = None) -> Resume:
    """One LLM call. The result is stored, so this never runs per posting."""
    resume, _ = call(
        "intake",
        [{"role": "system", "content": SYSTEM},
         # Fenced and labelled as data for the same reason job adverts are: a
         # resume is a document the model reads, not a source of instructions,
         # and "ignore previous instructions" in white 1pt text is a known trick.
         {"role": "user", "content": f"<resume>\n{redact(text)}\n</resume>"}],
        Resume,
    )
    resume.source_file = source_file
    resume.parsed_on = today or date.today()
    return _trim(resume)


def _trim(resume: Resume, max_items: int = 12) -> Resume:
    """Enforce the size limits the prompt asks for but cannot guarantee.

    Truncation rather than validation: a model that writes 210 characters of
    headline has produced a usable resume with one long field, and rejecting it
    would throw the other nine away.
    """
    resume.headline = resume.headline[:200]
    resume.skills = resume.skills[:max_items]
    resume.tools = resume.tools[:max_items]
    resume.projects = resume.projects[:max_items]
    resume.education = resume.education[:4]
    resume.experience = resume.experience[:6]
    for job in resume.experience:
        job.highlights = job.highlights[:4]
    return resume


def from_pdf(path: Path) -> Resume:
    """extract -> redact -> parse, for callers that want the whole pipeline."""
    return parse(extract_text(path), source_file=Path(path).name)
