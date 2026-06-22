from __future__ import annotations

from langchain_core.messages import AIMessage

from ops_rag_agent.config import settings
from ops_rag_agent.guardrails import allow_guardrail_result, merge_guardrail_state, review_agent_guardrails
from ops_rag_agent.observability import append_audit_event
from ops_rag_agent.prompts import get_prompt_spec
from ops_rag_agent.rag.retriever import RagRetriever


def run_rag_agent(state: dict, retriever: RagRetriever) -> dict:
    query = state.get("user_query", "")
    prompt_spec = get_prompt_spec("rag.answering")
    rag_queries = retriever.build_queries(query)
    candidates = retriever.hybrid_retrieve(rag_queries)
    reranked = retriever.rerank(query, candidates)

    answer = (
        "RAG agent finished query expansion, BM25+dense hybrid retrieval, cache lookup, "
        "and cross-encoder rerank interface execution. Replace the internal adapters with "
        "Milvus sparse/dense search and a production reranker."
    )
    citations = [doc["source"] for doc in reranked]

    updates = {
        "rag_queries": rag_queries,
        "rag_candidates": candidates,
        "rag_reranked": reranked,
        "citations": citations,
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "workflow_status": "running",
        "approval_status": "not_required",
        "prompt_versions": {
            **state.get("prompt_versions", {}),
            "rag_agent": prompt_spec.version,
        },
    }
    guardrail_result = (
        review_agent_guardrails(
            route="rag",
            state=state,
            updates=updates,
            enable_model_review=settings.guardrails_model_review_enabled,
        )
        if settings.guardrails_enabled
        else allow_guardrail_result()
    )
    updates.update(merge_guardrail_state(state, guardrail_result))
    updates["audit_trail"] = append_audit_event(
        state,
        "rag_retrieval_completed",
        node="rag_agent",
        details={
            "query_count": len(rag_queries),
            "candidate_count": len(candidates),
            "reranked_count": len(reranked),
            "prompt_version": prompt_spec.version,
            "guardrail_action": updates.get("guardrail_action", "allow"),
        },
    )
    return updates
