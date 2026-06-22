"""Skill：知识库混合检索 + rerank。

把原 `RagRetriever` 包装成统一注册表里的 skill，让 LLM 在需要查阅资料/文档/
知识库时可以主动选择调用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from ops_rag_agent.observability import trace_skill_call
from ops_rag_agent.rag.retriever import RagRetriever
from ops_rag_agent.skills.base import SkillKind, SkillSpec


class RagSearchArgs(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class RagSearchResult(BaseModel):
    query: str
    top_k: int
    results: list[dict[str, Any]]


@dataclass
class RagSearchSkill:
    spec: SkillSpec = SkillSpec(
        skill_id="rag.search",
        name="Knowledge Base Search",
        description=(
            "对企业内部知识库（Milvus）进行 BM25 + 向量混合检索，并跑一次 rerank，"
            "返回 top_k 条带评分与来源的文档片段。"
        ),
        version="1.0.0",
        business_domain="knowledge",
        kind=SkillKind.REGULAR,
        requires_approval=False,
        is_readonly=True,
        timeout_s=15,
        tags=("rag", "kb", "search", "retrieval"),
        when_to_use=(
            "当用户问题需要查询内部文档/知识库/手册/Playbook/历史事故/FAQ 等结构化资料时使用。"
            "适合：查询某个组件的故障处理流程、配置规范、概念解释、SOP 等知识检索类需求。"
            "不适合：实时数据查询（用 Prometheus）、本机系统状态（用 ops.local.snapshot）、"
            "公网最新资讯（用 web.search）。"
        ),
        argument_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户原始问题或重写后的检索短语，建议 1-2 句话。",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回片段数量，默认 5，建议 3-8。",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        argument_model=RagSearchArgs,
        result_model=RagSearchResult,
        example_invocations=(
            {"query": "k8s pod CrashLoopBackOff 排查流程", "top_k": 5},
            {"query": "Milvus collection 索引参数推荐"},
        ),
        risk_level="low",
    )

    retriever: RagRetriever = field(default_factory=RagRetriever)

    @trace_skill_call("rag.search")
    def invoke(self, arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return "ERROR: Missing argument: query"
        top_k = int(arguments.get("top_k") or 5)
        if top_k <= 0 or top_k > 20:
            top_k = 5

        try:
            queries = self.retriever.build_queries(query)
            candidates = self.retriever.hybrid_retrieve(queries, top_k=top_k)
            reranked = self.retriever.rerank(query, candidates)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: rag_retrieval_failed: {type(exc).__name__}: {exc}"

        # 精简输出，避免 prompt 爆长
        compact = []
        for doc in reranked[:top_k]:
            compact.append(
                {
                    "doc_id": doc.get("doc_id"),
                    "chunk_id": doc.get("chunk_id"),
                    "score": doc.get("score"),
                    "source": doc.get("source"),
                    "text": str(doc.get("text", ""))[:600],
                    "channels": doc.get("metadata", {}).get("retrieval_channels", []),
                }
            )
        return json.dumps(
            {"query": query, "top_k": top_k, "results": compact},
            ensure_ascii=False,
        )
