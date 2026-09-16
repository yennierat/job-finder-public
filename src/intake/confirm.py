"""Render the profile as YAML and let the human accept or edit it.

This step plus inferred_fields is the entire defence against the intake model
quietly inventing constraints nobody asked for. Do not skip either.
"""

import os
import subprocess
import tempfile

import yaml

from src.models import JobProfile


def to_yaml_obj(model) -> str:
    """Any pydantic model as YAML — used to show a parsed resume for review."""
    return yaml.safe_dump(model.model_dump(mode="json", exclude_none=False),
                          sort_keys=False, allow_unicode=True)


def to_yaml(profile: JobProfile) -> str:
    return to_yaml_obj(profile)


def show(profile: JobProfile) -> None:
    print("\n" + "=" * 60)
    print(to_yaml(profile))
    if profile.inferred_fields:
        print("These were INFERRED, not stated by you — check them:")
        for field in profile.inferred_fields:
            print(f"  - {field}: {getattr(profile, field, '(unknown field)')}")
    print("=" * 60)


def edit(profile: JobProfile) -> JobProfile:
    """Open the YAML in $EDITOR and re-validate whatever comes back."""
    editor = os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "vi")
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False,
                                     encoding="utf-8") as f:
        f.write(to_yaml(profile))
        path = f.name
    try:
        subprocess.call([editor, path])
        with open(path, encoding="utf-8") as f:
            return JobProfile.model_validate(yaml.safe_load(f.read()))
    finally:
        os.unlink(path)


def save(profile: JobProfile, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml(profile), encoding="utf-8")
