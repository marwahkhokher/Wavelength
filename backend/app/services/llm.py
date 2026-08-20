from langchain_openai import ChatOpenAI

from app.config import settings


def get_llm(temperature: float = 0.4) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
    )
