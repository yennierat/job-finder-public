"""Free text -> JobProfile draft, with every field tagged stated or inferred."""

from datetime import date

from pydantic import BaseModel, Field

from src.llm import call
from src.models import JobProfile

SYSTEM = """You convert a person's description of the jobs they want into a structured profile.

Return ONLY JSON with these fields:
{
  "employment_types": [one or more of "internship","full_time","contract","graduate_programme"],
  "locations": [canonical city or country names, e.g. "Singapore", "Hong Kong"],
  "remote_ok": bool,
  "start_window": {"start":"YYYY-MM-DD","end":"YYYY-MM-DD"} or null,
  "fields": [domains/industries, e.g. "fintech", "trading", "infrastructure"],
  "seniority": one of "intern","entry","mid" or null,
  "must_have_keywords": [words that should appear in a title; keep this SHORT and broad],
  "exclude_keywords": [words that disqualify a title],
  "freeform_notes": "<=500 chars of anything else that matters",
  "inferred_fields": [names of fields you guessed rather than were told]
}

Critical rules:
- Resolve all relative dates ("next summer", "this winter") to absolute dates using TODAY'S DATE below. Never emit relative language.
- List in inferred_fields EVERY field you filled in that the user did not explicitly state. Be honest and complete: this list is shown to the user for confirmation.
- Keep must_have_keywords broad. Narrow keywords silently destroy recall.
"""


class ProfileDraft(BaseModel):
    employment_types: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    remote_ok: bool = False
    start_window: dict | None = None
    fields: list[str] = Field(default_factory=list)
    seniority: str | None = None
    must_have_keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)
    freeform_notes: str = ""
    inferred_fields: list[str] = Field(default_factory=list)


def extract(text: str, today: date | None = None) -> ProfileDraft:
    today = today or date.today()
    draft, _ = call(
        "intake",
        [{"role": "system", "content": f"{SYSTEM}\n\nTODAY'S DATE: {today.isoformat()}"},
         {"role": "user", "content": text}],
        ProfileDraft,
    )
    return draft


def to_profile(draft: ProfileDraft) -> JobProfile:
    """Validate the draft into a real JobProfile (raises on bad data)."""
    payload = draft.model_dump()
    payload["freeform_notes"] = (payload.get("freeform_notes") or "")[:500]
    return JobProfile.model_validate(payload)
