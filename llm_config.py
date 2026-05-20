from langchain_openai import ChatOpenAI
from openai import (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)

# Best → lightest
FALLBACK_MODELS = [
    "gpt-5-mini",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]

def create_llm(model: str, **kwargs):
    return ChatOpenAI(
        model=model,
        temperature=0,
        **kwargs
    )

def build_fallback_llm():
    """
    Creates fallback chain:
    gpt-5-mini → gpt-5.4-mini → gpt-5.4-nano
    """

    llms = [create_llm(m) for m in FALLBACK_MODELS]

    primary = llms[0]
    fallbacks = llms[1:]

    return primary.with_fallbacks(
        fallbacks,
        exceptions_to_handle=(
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ),
    )
