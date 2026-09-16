"""Shared data models for the internship monitor."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

EmploymentType = Literal["internship", "full_time", "contract", "graduate_programme"]
Seniority = Literal["intern", "entry", "mid"]

PROFILE_VERSION = "1"


class DateRange(BaseModel):
    start: date
    end: date

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: date, info):
        start = info.data.get("start")
        if start and v < start:
            raise ValueError("end must not precede start")
        return v


class ResumeExperience(BaseModel):
    role: str = ""
    organisation: str = ""
    period: str = ""  # free text as written on the CV: "Jun 2025 - Aug 2025"
    highlights: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    """A resume reduced to the parts that bear on whether a job fits.

    Deliberately small. This is rendered into EVERY classification prompt, so it
    is a summary, not the document: a 4,000-character CV pasted into a 6-posting
    batch would crowd out the job adverts the model is meant to be reading.
    Parsed once by `tools/import_resume.py`, then stored in profile.yaml.

    Contact details are never a field here — see resume.redact().
    """

    # No max_length: pydantic would REJECT an over-long headline, failing the
    # whole resume parse and sending the chain to the next model over a field
    # that only needed trimming. resume._trim() truncates instead.
    headline: str = ""
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    experience: list[ResumeExperience] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    # Internships and part-time work count; this is "how much industry exposure",
    # not "years since graduation".
    years_experience: float | None = None
    source_file: str = ""
    parsed_on: date | None = None

    def is_empty(self) -> bool:
        return not (self.skills or self.tools or self.experience
                    or self.projects or self.headline)


class JobProfile(BaseModel):
    employment_types: list[EmploymentType]
    locations: list[str] = Field(default_factory=list)
    remote_ok: bool = False
    start_window: DateRange | None = None
    fields: list[str] = Field(default_factory=list)
    seniority: Seniority | None = None
    must_have_keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)
    source_tags: list[str] | None = None
    freeform_notes: str = Field(default="", max_length=500)
    inferred_fields: list[str] = Field(default_factory=list)
    # Optional. Absent means classification judges the search criteria only, and
    # verdicts carry no fit_score — the feature is additive, never required.
    resume: Resume | None = None
    profile_version: str = PROFILE_VERSION


class Posting(BaseModel):
    """One job posting, normalised across ATS platforms."""

    source_id: str
    external_id: str
    title: str
    url: str | None = None
    location_raw: str | None = None
    location: str | None = None  # canonicalised in normalise()
    employment_type: EmploymentType | None = None
    remote: bool | None = None
    content_hash: str = ""
    # Plain-text job description where the ATS returns one in its list response.
    # Not persisted: it is large, and it is only needed for the one classify call
    # a posting ever gets.
    description: str | None = None
    # Application deadline, ISO date. Either a structured field from the ATS
    # (Workday exposes one) or matched out of the advert text by rule — never
    # produced by the model, which would invent plausible dates. deadline_text
    # keeps the sentence it came from so any date can be checked against source.
    deadline: str | None = None
    deadline_text: str | None = None


WorkAuthorization = Literal[
    "not_mentioned",     # nothing in the supplied text either way
    "sponsorship_offered",
    "citizen_or_pr_required",
    "authorization_required",  # must already have the right to work, no sponsorship
    "unclear",
]


class Verdict(BaseModel):
    """One classification result for a single posting."""

    id: str  # echoes Posting.external_id — never match verdicts positionally
    is_match: bool
    category: str = ""
    reason: str = ""
    work_authorization: WorkAuthorization = "not_mentioned"
    # Must be verbatim from the posting text. Empty whenever no supporting text
    # was supplied — an invented quote is worse than no quote.
    authorization_quote: str = ""

    # --- resume fit. None whenever no resume is configured; the prompt does not
    # ask for these fields at all in that case, and nothing downstream may treat
    # a missing score as a low one.
    fit_score: int | None = None
    # Skills the posting asks for that the resume evidences, and asks for that it
    # does not. Requiring gaps is what stops the model scoring everything 90:
    # it has to name something missing before it can claim a high number.
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)

    @field_validator("fit_score")
    @classmethod
    def clamp_score(cls, v: int | None) -> int | None:
        """Models emit 8.5, 850 and "95%" for a 0-100 field. Clamp, never reject:
        losing an otherwise good verdict over a stray percentage sign would be a
        worse outcome than a slightly wrong number."""
        if v is None:
            return None
        return max(0, min(100, int(v)))


class VerdictBatch(BaseModel):
    results: list[Verdict]


class CallMeta(BaseModel):
    model: str
    provider: str = ""
    attempt: int
    fallback_depth: int
    latency_ms: int
    input_chars: int
    output_chars: int
    finish_reason: str = ""
    session_id: str = ""
    generation_id: str = ""  # OpenRouter's own id for this generation
