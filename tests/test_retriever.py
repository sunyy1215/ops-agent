from ops_rag_agent.rag.retriever import RagRetriever


def test_hybrid_retrieve_marks_cache_hits_on_repeat_queries() -> None:
    retriever = RagRetriever()
    queries = retriever.build_queries("k8s cpu alert")

    first = retriever.hybrid_retrieve(queries, top_k=4)
    second = retriever.hybrid_retrieve(queries, top_k=4)

    assert first
    assert first[0]["metadata"]["cache_status"] == "miss"
    assert second[0]["metadata"]["cache_status"] == "hit"
    assert second[0]["metadata"]["cache_key"].startswith("retrieval:")


def test_rerank_adds_cross_encoder_metadata() -> None:
    retriever = RagRetriever()
    queries = retriever.build_queries("service memory spike")
    docs = retriever.hybrid_retrieve(queries, top_k=4)

    reranked = retriever.rerank("service memory spike", docs)

    assert reranked
    assert reranked[0]["metadata"]["rerank_strategy"] == "cross_encoder_interface"
    assert "cross_encoder_score" in reranked[0]["metadata"]
