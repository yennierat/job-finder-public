"""Check whether the configured OpenRouter model is a free one.

Reads OPENROUTER_API_KEY (optional) and MODEL from .env, fetches the
current model list from OpenRouter, and reports whether MODEL is free.
"""

import json
import os
import urllib.request

MODELS_URL = "https://openrouter.ai/api/v1/models"
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")


def load_env(path: str = ENV_PATH) -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_free_models() -> list[str]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(MODELS_URL, headers=headers)
    with urllib.request.urlopen(request) as response:
        data = json.load(response)

    return [
        model["id"]
        for model in data["data"]
        if float(model["pricing"]["prompt"]) == 0
        and float(model["pricing"]["completion"]) == 0
    ]


def is_free_model(model_id: str, free_models: list[str] | None = None) -> bool:
    free_models = free_models if free_models is not None else get_free_models()
    return model_id in free_models


def main() -> None:
    load_env()
    model = os.environ.get("MODEL")
    if not model:
        print("MODEL is not set in .env")
        return

    free_models = get_free_models()
    if is_free_model(model, free_models):
        print(f"{model} is a free model.")
    else:
        print(f"{model} is NOT a free model.")


if __name__ == "__main__":
    main()
