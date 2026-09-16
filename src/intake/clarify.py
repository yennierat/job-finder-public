"""Targeted follow-up questions for the fields the model was least sure about."""

from pydantic import BaseModel, Field

from src.intake.extract import ProfileDraft
from src.llm import call

MAX_QUESTIONS = 4

SYSTEM = """You are refining a draft job-search profile by asking the user a few questions.

Return ONLY JSON: {"questions": ["...", "..."]}

Ask at most 4 short questions, and only about things that would actually change
which jobs match: fields the draft guessed at (see inferred_fields), missing
locations or dates, or a scope that is far too broad ("anything in tech") or too
narrow to return results. Ask nothing if the draft is already workable — return
an empty list."""

REVISE_SYSTEM = """Update the draft profile using the user's answers.

Return ONLY JSON in the same shape as the draft you were given. Keep
inferred_fields accurate: remove fields the user has now confirmed explicitly,
and keep listing anything still guessed."""


class Questions(BaseModel):
    questions: list[str] = Field(default_factory=list)


def ask(draft: ProfileDraft) -> list[str]:
    result, _ = call(
        "intake",
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": draft.model_dump_json(indent=2)}],
        Questions,
    )
    return result.questions[:MAX_QUESTIONS]


def revise(draft: ProfileDraft, qa: list[tuple[str, str]]) -> ProfileDraft:
    transcript = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa if a.strip())
    if not transcript:
        return draft
    revised, _ = call(
        "intake",
        [{"role": "system", "content": REVISE_SYSTEM},
         {"role": "user", "content": f"Draft:\n{draft.model_dump_json(indent=2)}\n\n"
                                     f"Answers:\n{transcript}"}],
        ProfileDraft,
    )
    return revised
