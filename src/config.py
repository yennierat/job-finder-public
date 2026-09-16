"""Loads and validates profile.yaml / sources.yaml, and reads .env."""

import hashlib
import json
import os
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.models import JobProfile

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
PROFILE_PATH = CONFIG_DIR / "profile.yaml"
SOURCES_PATH = CONFIG_DIR / "sources.yaml"
DB_PATH = ROOT / "state.db"


def load_env(path: Path | None = None) -> None:
    """Minimal .env reader — real environment always wins."""
    path = path or (ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Source(BaseModel):
    id: str
    name: str
    platform: str
    board: str
    tags: list[str] = []
    # For boards whose every posting is in one place but whose location strings
    # are internal site names ("SLA-REVENUE HOUSE"). Applied only where
    # canonicalisation found nothing, so it never overrides a real location.
    default_location: str | None = None


def profile_hash(profile: JobProfile) -> str:
    """Stable hash of the profile's semantic content.

    inferred_fields is excluded — it is provenance metadata about how the profile
    was produced, not a filtering constraint, so changing it must not invalidate
    perfectly good cached verdicts.
    """
    payload = profile.model_dump(mode="json", exclude={"inferred_fields"})
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_profile(path: Path | None = None) -> JobProfile:
    path = path or PROFILE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.intake` to create it"
        )
    return JobProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_sources(path: Path | None = None,
                 tags: list[str] | None = None) -> list[Source]:
    path = path or SOURCES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python tools/discover_boards.py` to create it"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = [Source.model_validate(s) for s in raw.get("sources", [])]
    if tags:
        wanted = set(tags)
        sources = [s for s in sources if wanted & set(s.tags)]
    return sources
