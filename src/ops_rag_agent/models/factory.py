from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ops_rag_agent.config import settings


def build_router_llm(*, native_tools: Optional[list[dict]] = None) -> ChatOpenAI:
    llm = ChatOpenAI(
        model=settings.llm_model_router,
        api_key=settings.llm_api_key,
        base_url=settings.llm_api_base,
        temperature=0,
    )
    if native_tools:
        return llm.bind_tools(native_tools)
    return llm


def build_chat_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model_chat,
        api_key=settings.llm_api_key,
        base_url=settings.llm_api_base,
        temperature=0.2,
    )


def build_reasoning_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model_reasoning,
        api_key=settings.llm_api_key,
        base_url=settings.llm_api_base,
        temperature=0,
    )


def build_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key or settings.llm_api_key,
        base_url=settings.embedding_api_base or settings.llm_api_base,
    )
