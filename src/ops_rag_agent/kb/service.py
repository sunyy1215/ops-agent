from __future__ import annotations

from typing import Any

from ops_rag_agent.config import settings
from ops_rag_agent.rag.retriever import RagRetriever


def search_knowledge(
    query: str,
    *,
    fused_top_k: int | None = None,
    rerank_top_k: int | None = None,
    metadata_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    retriever = RagRetriever()
    queries = retriever.build_queries(query)
    docs = retriever.hybrid_retrieve(
        queries,
        top_k=fused_top_k or settings.retrieval_fused_top_k,
    )
    reranked = retriever.rerank(query, docs)
    limited = reranked[: rerank_top_k or settings.rerank_top_k]
    return {
        "query": query,
        "rewritten_queries": queries,
        "candidate_count": len(docs),
        "reranked_count": len(limited),
        "metadata_filters": metadata_filters or {},
        "citations": [str(item.get("source") or item.get("doc_id") or "") for item in limited],
        "results": limited,
    }
