from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI


def create_llm(model: str, **kwargs):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        **kwargs,
    )
