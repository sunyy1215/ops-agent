from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ops_rag_agent.cache import (
    CacheKeyBuilder,
    CacheNamespace,
    CompositeRetrievalCache,
    InMemoryRetrievalCache,
    RedisRetrievalCache,
)
from ops_rag_agent.config import settings


@dataclass(frozen=True)
class RetrievalPlan:
    query: str
    query_variants: list[str]
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    fusion_strategy: str = "weighted_rrf"


@dataclass
class RagRetriever:
    """BM25 + vector hybrid retrieval skeleton with cross-encoder rerank hooks."""

    cache: CompositeRetrievalCache = field(
        default_factory=lambda: CompositeRetrievalCache(
            memory_cache=InMemoryRetrievalCache(),
            redis_cache=RedisRetrievalCache(settings.retrieval_cache_redis_url),
        )
    )

    def build_queries(self, user_query: str) -> list[str]:
        query = user_query.strip()
        if not query:
            return []
        return [query, f"keyword: {query}", f"ops-docs: {query}"]

    def hybrid_retrieve(self, queries: list[str], top_k: int = 8) -> list[dict]:
        if not queries:
            return []
        plan = RetrievalPlan(query=queries[0], query_variants=queries)
        cache_payload = {
            "collection": settings.milvus_collection,
            "queries": queries,
            "top_k": top_k,
            "bm25_weight": plan.bm25_weight,
            "vector_weight": plan.vector_weight,
            "fusion": plan.fusion_strategy,
            "ttl_seconds": settings.retrieval_cache_ttl_seconds,
        }
        cache_key = CacheKeyBuilder.build(
            CacheNamespace(layer="retrieval", name=settings.retrieval_cache_namespace),
            cache_payload,
        )

        cached = self.cache.get(cache_key.key)
        if cached is not None:
            return [self._mark_cache_hit(doc, cache_key.key) for doc in cached]

        bm25_hits = self._bm25_retrieve(plan, top_k=top_k)
        vector_hits = self._vector_retrieve(plan, top_k=top_k)
        fused = self._fuse_results(
            bm25_hits,
            vector_hits,
            bm25_weight=plan.bm25_weight,
            vector_weight=plan.vector_weight,
            top_k=top_k,
        )
        self.cache.set(cache_key.key, fused, ttl_seconds=cache_key.ttl_seconds)
        return fused

    def rerank(self, user_query: str, docs: list[dict]) -> list[dict]:
        reranked: list[dict[str, Any]] = []
        for index, doc in enumerate(docs):
            cross_encoder_score = self._cross_encoder_score(user_query, doc)
            metadata = dict(doc.get("metadata", {}))
            metadata.update(
                {
                    "rerank_strategy": "cross_encoder_interface",
                    "rerank_model": settings.rerank_model or "placeholder-cross-encoder",
                    "initial_rank": index,
                    "cross_encoder_score": cross_encoder_score,
                }
            )
            reranked.append(
                {
                    **doc,
                    "score": round((float(doc.get("score", 0.0)) * 0.3) + cross_encoder_score * 0.7, 4),
                    "metadata": metadata,
                }
            )

        reranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return reranked[: settings.rerank_top_k]

    def _bm25_retrieve(self, plan: RetrievalPlan, top_k: int) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for index, query in enumerate(plan.query_variants[:2], start=1):
            docs.append(
                {
                    "doc_id": f"hybrid-doc-{index}",
                    "chunk_id": f"bm25-chunk-{index}",
                    "score": round(0.9 - index * 0.08, 4),
                    "source": "milvus://" + settings.milvus_collection,
                    "text": f"BM25 placeholder hit for '{query}'. Replace with sparse/BM25 search adapter.",
                    "metadata": {
                        "retrieval_channel": "bm25",
                        "query": query,
                        "top_k": top_k,
                    },
                }
            )
        return docs[:top_k]

    def _vector_retrieve(self, plan: RetrievalPlan, top_k: int) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for index, query in enumerate(plan.query_variants[:2], start=1):
            docs.append(
                {
                    "doc_id": f"hybrid-doc-{index}",
                    "chunk_id": f"vector-chunk-{index}",
                    "score": round(0.92 - index * 0.05, 4),
                    "source": "milvus://" + settings.milvus_collection,
                    "text": f"Vector placeholder hit for '{query}'. Replace with embedding + Milvus dense search.",
                    "metadata": {
                        "retrieval_channel": "vector",
                        "query": query,
                        "top_k": top_k,
                    },
                }
            )
        return docs[:top_k]

    def _fuse_results(
        self,
        bm25_hits: list[dict[str, Any]],
        vector_hits: list[dict[str, Any]],
        *,
        bm25_weight: float,
        vector_weight: float,
        top_k: int,
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for channel, weight, docs in (
            ("bm25", bm25_weight, bm25_hits),
            ("vector", vector_weight, vector_hits),
        ):
            for rank, doc in enumerate(docs, start=1):
                key = str(doc["doc_id"])
                existing = fused.get(key, {**doc, "metadata": dict(doc.get("metadata", {}))})
                hybrid_score = existing.get("hybrid_score", 0.0) + (weight / (50 + rank))
                channels = set(existing["metadata"].get("retrieval_channels", []))
                channels.add(channel)
                existing["hybrid_score"] = round(hybrid_score, 6)
                existing["score"] = round(max(float(existing.get("score", 0.0)), float(doc.get("score", 0.0))), 4)
                existing["metadata"]["retrieval_channels"] = sorted(channels)
                existing["metadata"]["fusion_strategy"] = "weighted_rrf"
                fused[key] = existing

        ranked = sorted(fused.values(), key=lambda item: item.get("hybrid_score", 0.0), reverse=True)
        for item in ranked:
            metadata = dict(item.get("metadata", {}))
            metadata["cache_status"] = "miss"
            item["metadata"] = metadata
        return ranked[:top_k]

    @staticmethod
    def _cross_encoder_score(user_query: str, doc: dict[str, Any]) -> float:
        query_terms = set(user_query.lower().split())
        text_terms = set(str(doc.get("text", "")).lower().split())
        overlap = len(query_terms & text_terms)
        normalized = overlap / max(len(query_terms), 1)
        base_score = float(doc.get("hybrid_score", doc.get("score", 0.0)))
        return round(min(1.0, base_score + normalized * 0.2), 4)

    @staticmethod
    def _mark_cache_hit(doc: dict[str, Any], cache_key: str) -> dict[str, Any]:
        metadata = dict(doc.get("metadata", {}))
        metadata.update({"cache_status": "hit", "cache_key": cache_key})
        return {**doc, "metadata": metadata}
