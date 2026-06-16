from langchain_openai import ChatOpenAI


def create_llm(model: str = "gpt-5-mini", **kwargs):
    return ChatOpenAI(model=model, temperature=0, **kwargs)


# from langchain_google_genai import ChatGoogleGenerativeAI
# from google.api_core.exceptions import (
#     ResourceExhausted,
#     DeadlineExceeded,
#     InternalServerError,
#     ServiceUnavailable,
# )

# # Best → lightest
# FALLBACK_MODELS = [
#     "gemini-2.5-pro",
#     "gemini-2.5-flash",
#     "gemini-2.5-flash-lite",
# ]


# def create_llm(model: str, **kwargs):
#     return ChatGoogleGenerativeAI(
#         model=model,
#         temperature=0,
#         google_api_key="YOUR_GOOGLE_API_KEY",
#         **kwargs
#     )


# def build_fallback_llm():
#     """
#     Creates fallback chain:
#     gemini-2.5-pro
#         ↓
#     gemini-2.5-flash
#         ↓
#     gemini-2.5-flash-lite
#     """

#     llms = [create_llm(m) for m in FALLBACK_MODELS]

#     primary = llms[0]
#     fallbacks = llms[1:]

#     return primary.with_fallbacks(
#         fallbacks,
#         exceptions_to_handle=(
#             ResourceExhausted,     # quota/rate limit
#             DeadlineExceeded,      # timeout
#             ServiceUnavailable,
#             InternalServerError,
#         ),
#     )


