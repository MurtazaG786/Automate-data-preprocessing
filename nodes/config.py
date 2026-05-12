from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded,
    InternalServerError,
)

from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAIError
)

FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


def create_llm(model: str, **kwargs):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        **kwargs,
    )


def build_fallback_llm():

    llms = [create_llm(model) for model in FALLBACK_MODELS]

    primary = llms[0]

    fallbacks = llms[1:]

    return primary.with_fallbacks(
        fallbacks,
        exceptions_to_handle=(
            ResourceExhausted,
            ServiceUnavailable,
            DeadlineExceeded,
            InternalServerError,
            ChatGoogleGenerativeAIError,
        ),
    )