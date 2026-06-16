import os
from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_DOTENV_PATH = os.path.join(_REPO_ROOT, ".env")
load_dotenv(dotenv_path=_DOTENV_PATH, override=True)


def get_primary_api_key_model() -> tuple[str | None, str | None]:
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("MODEL_NAME")

    if api_key:
        api_key = api_key.strip()
    if model_name:
        model_name = model_name.strip()

    return (api_key or None, model_name or None)
