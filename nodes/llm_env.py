import os
from typing import List, Tuple

from dotenv import load_dotenv

# Load .env from repo root so Streamlit or other cwd changes still work.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_DOTENV_PATH = os.path.join(_REPO_ROOT, ".env")
load_dotenv(dotenv_path=_DOTENV_PATH, override=True)


def get_api_keys_and_models() -> Tuple[List[str], List[str]]:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()

    keys = [api_key] if api_key else []
    models = [model_name] if model_name else []

    return keys, models


def get_primary_api_key_model() -> Tuple[str | None, str | None]:
    keys, models = get_api_keys_and_models()
    if not keys or not models:
        return None, None
    return keys[0], models[0]
