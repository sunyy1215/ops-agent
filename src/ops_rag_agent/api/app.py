from __future__ import annotations

from collections import Counter
import re
import uuid
from functools import lru_cache
from typing import Any
from typing import Optional

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field
from pydantic import ValidationError
from pydantic import model_validator

from ops_rag_agent.config import settings
from ops_rag_agent.graph.app import build_graph
from ops_rag_agent.graph.checkpointing import build_checkpointer
from ops_rag_agent.kb.ingestion.pipeline import ingest_directory, ingest_file, ingest_inline_documents
from ops_rag_agent.kb.service import search_knowledge
from ops_rag_agent.memory.store import InMemoryLongTermMemoryStore, LongTermMemoryRecord, MilvusLongTermMemoryStore
from ops_rag_agent.observability import graph_tracing_context
from ops_rag_agent.skills.bootstrap import build_skill_registry
from ops_rag_agent.skills.runtime import build_runtime_summary


class InvokeRequest(BaseModel):
    user_query: Optional[str] = Field(default=None, min_length=1)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    thread_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    resume: Any = None

    @model_validator(mode="after")
    def validate_invoke_request(self) -> "InvokeRequest":
        if not self.user_query and self.resume is None:
            raise ValueError("user_query and resume cannot both be empty")
        return self


class InvokeResponse(BaseModel):
    thread_id: str
    checkpoint_id: Optional[str] = None
    route: Optional[str] = None
    final_answer: str
    workflow_status: str = "completed"
    approval_required: bool = False
    approval_status: Optional[str] = None
    pending_interrupts: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
    ops_hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    ops_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    ops_failed_skills: list[dict[str, Any]] = Field(default_factory=list)
    ops_skill_calls: list[dict[str, Any]] = Field(default_factory=list)
    ops_execution_steps: list[dict[str, Any]] = Field(default_factory=list)
    router_scratchpad: list[dict[str, Any]] = Field(default_factory=list)
    router_iterations: int = 0
    router_stop_reason: str = ""
    router_validation_results: list[dict[str, Any]] = Field(default_factory=list)
    router_runtime_observations: list[dict[str, Any]] = Field(default_factory=list)
    runtime_audit_records: list[dict[str, Any]] = Field(default_factory=list)
    runtime_events: list[dict[str, Any]] = Field(default_factory=list)
    runtime_summary: dict[str, Any] = Field(default_factory=dict)
    intent_analysis: dict[str, Any] = Field(default_factory=dict)
    execution_plan: dict[str, Any] = Field(default_factory=dict)
    plan_progress: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)
    rolling_summary: str = ""


class KbDocument(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class KbIngestRequest(BaseModel):
    collection: Optional[str] = None
    file_path: Optional[str] = None
    directory_path: Optional[str] = None
    id_prefix: str = "api-"
    glob: str = "**/*"
    dry_run: bool = False
    docs: list[KbDocument] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_payload(self) -> "KbIngestRequest":
        provided = sum(
            1 for item in [bool(self.file_path), bool(self.directory_path), bool(self.docs)] if item
        )
        if provided != 1:
            raise ValueError("Exactly one of file_path, directory_path, or docs must be provided.")
        return self


class KbIngestResponse(BaseModel):
    collection: str
    inserted_count: int
    ids: list[str]
    run_id: Optional[str] = None
    dry_run: bool = False
    stats: dict[str, Any] = Field(default_factory=dict)
    duplicate_relations: list[dict[str, str]] = Field(default_factory=list)


class KbSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    fused_top_k: Optional[int] = None
    rerank_top_k: Optional[int] = None
    visibility: Optional[str] = None
    biz_domain: Optional[str] = None
    owner: Optional[str] = None
    lang: Optional[str] = None
    source: Optional[str] = None
    source_id: Optional[str] = None
    doc_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class KbSearchResponse(BaseModel):
    query: str
    rewritten_queries: list[str]
    candidate_count: int
    reranked_count: int
    metadata_filters: dict[str, Any]
    citations: list[str]
    results: list[dict[str, Any]]


class MemoryWriteItem(BaseModel):
    memory_id: str
    memory_type: str
    content: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryWriteRequest(BaseModel):
    records: list[MemoryWriteItem]


class MemoryWriteResponse(BaseModel):
    written_ids: list[str]
    count: int


class MemoryRecallRequest(BaseModel):
    query: str = Field(min_length=1)
    user_id: Optional[str] = None
    top_k: int = 3


class MemoryRecallResponse(BaseModel):
    query: str
    count: int
    results: list[dict[str, Any]]


class RuntimeConfigUpdateRequest(BaseModel):
    updates: dict[str, Any] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def _memory_store() -> InMemoryLongTermMemoryStore | MilvusLongTermMemoryStore:
    if settings.long_term_memory_backend == "milvus":
        return MilvusLongTermMemoryStore()
    return InMemoryLongTermMemoryStore()


SENSITIVE_CONFIG_FIELDS = {
    "llm_api_key",
    "embedding_api_key",
    "rerank_api_key",
    "milvus_token",
    "prometheus_auth_token",
    "langsmith_api_key",
}

RUNTIME_EDITABLE_CONFIG_FIELDS = (
    "llm_model_router",
    "llm_model_chat",
    "llm_model_reasoning",
    "embedding_model",
    "rerank_model",
    "rerank_top_k",
    "rerank_backend",
    "rag_vector_backend",
    "long_term_memory_backend",
    "rag_bm25_backend",
    "retrieval_bm25_weight",
    "retrieval_vector_weight",
    "retrieval_bm25_top_k",
    "retrieval_ann_top_k",
    "retrieval_fused_top_k",
    "retrieval_rrf_k",
    "long_term_memory_top_k",
    "dialog_rag_first_enabled",
    "dialog_web_search_enabled",
    "dialog_rag_match_min_score",
    "dialog_web_search_max_results",
    "dialog_web_search_timeout_s",
    "route_llm_router_enabled",
    "router_native_tool_calling_enabled",
    "guardrails_enabled",
    "guardrails_model_review_enabled",
    "max_ops_workers",
    "compression_trigger_ratio",
    "max_compression_retries",
    "prometheus_base_url",
    "remote_ssh_default_host",
    "remote_ssh_port",
    "remote_ssh_connect_timeout_s",
    "remote_ssh_command_timeout_s",
)

PUBLIC_CONFIG_SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "application": ("app_name", "environment"),
    "llm": (
        "llm_api_base",
        "llm_model_router",
        "llm_model_chat",
        "llm_model_reasoning",
    ),
    "embedding": (
        "embedding_api_base",
        "embedding_model",
    ),
    "rag": (
        "rerank_model",
        "rerank_top_k",
        "rerank_backend",
        "retrieval_bm25_weight",
        "retrieval_vector_weight",
        "retrieval_bm25_top_k",
        "retrieval_ann_top_k",
        "retrieval_fused_top_k",
        "retrieval_rrf_k",
        "dialog_rag_first_enabled",
        "dialog_web_search_enabled",
        "dialog_rag_match_min_score",
        "dialog_web_search_max_results",
        "dialog_web_search_timeout_s",
        "route_llm_router_enabled",
        "router_native_tool_calling_enabled",
    ),
    "storage": (
        "milvus_uri",
        "milvus_collection",
        "memory_collection",
        "rag_vector_backend",
        "long_term_memory_backend",
        "rag_bm25_backend",
        "long_term_memory_top_k",
    ),
    "ops": (
        "prometheus_base_url",
        "remote_ssh_default_host",
        "remote_ssh_port",
        "remote_ssh_connect_timeout_s",
        "remote_ssh_command_timeout_s",
        "max_ops_workers",
    ),
    "guardrails": (
        "guardrails_enabled",
        "guardrails_model_review_enabled",
        "langsmith_tracing_enabled",
        "langsmith_project",
    ),
    "memory": (
        "compression_trigger_ratio",
        "max_compression_retries",
    ),
}


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "app_name": settings.app_name, "environment": settings.environment}

    @app.get("/config/public")
    def get_public_config() -> dict[str, Any]:
        return _build_public_config_payload()

    @app.put("/config/runtime")
    def update_runtime_config(request: RuntimeConfigUpdateRequest) -> dict[str, Any]:
        updated, rejected = _apply_runtime_config_updates(request.updates)
        return {
            "updated": updated,
            "rejected": rejected,
            "applied_count": len(updated),
            "public_config": _build_public_config_payload(),
        }

    @app.post("/invoke", response_model=InvokeResponse)
    def invoke(request: InvokeRequest) -> InvokeResponse:
        state = dict(request.initial_state)
        thread_id = request.thread_id or state.get("thread_id") or str(uuid.uuid4())
        state["thread_id"] = thread_id
        if request.user_query:
            state["user_query"] = request.user_query

        config = {"configurable": {"thread_id": thread_id}}
        if request.checkpoint_id:
            config["configurable"]["checkpoint_id"] = request.checkpoint_id

        graph = build_graph()
        with graph_tracing_context(thread_id=thread_id, resume=request.resume is not None):
            if request.resume is None:
                result = graph.invoke(state, config=config)
            else:
                resume_command = Command(
                    resume=request.resume,
                    update=state if state else None,
                )
                result = graph.invoke(resume_command, config=config)

        snapshot = graph.get_state(config)
        snapshot_values = _snapshot_state_values(snapshot)
        interrupts = _extract_snapshot_interrupts(snapshot, snapshot_values)
        final_answer = _extract_final_answer(snapshot_values if snapshot_values else result)
        return InvokeResponse(
            thread_id=thread_id,
            checkpoint_id=snapshot.config.get("configurable", {}).get("checkpoint_id"),
            route=snapshot_values.get("route") or result.get("route"),
            final_answer=final_answer,
            workflow_status=snapshot_values.get("workflow_status", "paused" if interrupts else "completed"),
            approval_required=bool(snapshot_values.get("approval_required", False)),
            approval_status=snapshot_values.get("approval_status"),
            pending_interrupts=interrupts,
            citations=list(snapshot_values.get("citations", []) or result.get("citations", [])),
            trace={
                "enabled": settings.langsmith_tracing_enabled,
                "project": settings.langsmith_project,
            },
            ops_hypotheses=list(snapshot_values.get("ops_hypotheses", [])),
            ops_recommendations=list(snapshot_values.get("ops_recommendations", [])),
            ops_failed_skills=list(snapshot_values.get("ops_failed_skills", [])),
            ops_skill_calls=list(snapshot_values.get("ops_skill_calls", [])),
            ops_execution_steps=list(snapshot_values.get("ops_execution_steps", [])),
            router_scratchpad=list(snapshot_values.get("router_scratchpad", [])),
            router_iterations=int(snapshot_values.get("router_iterations", 0) or 0),
            router_stop_reason=str(snapshot_values.get("router_stop_reason", "") or ""),
            router_validation_results=list(snapshot_values.get("router_validation_results", []) or []),
            router_runtime_observations=list(
                snapshot_values.get("router_runtime_observations", []) or []
            ),
            runtime_audit_records=list(snapshot_values.get("runtime_audit_records", []) or []),
            runtime_events=list(snapshot_values.get("runtime_events", []) or []),
            runtime_summary=dict(snapshot_values.get("runtime_summary", {}) or {}),
            intent_analysis=dict(snapshot_values.get("intent_analysis", {}) or {}),
            execution_plan=dict(snapshot_values.get("execution_plan", {}) or {}),
            plan_progress=list(snapshot_values.get("plan_progress", []) or []),
            token_usage=_build_token_usage(snapshot_values),
            rolling_summary=str(snapshot_values.get("rolling_summary", "") or ""),
        )

    @app.post("/kb/ingest", response_model=KbIngestResponse)
    def kb_ingest(request: KbIngestRequest) -> KbIngestResponse:
        if request.file_path:
            result = ingest_file(request.file_path, collection_name=request.collection, dry_run=request.dry_run)
            return KbIngestResponse(
                collection=request.collection or settings.milvus_collection,
                inserted_count=int(result["stats"].get("chunks_upserted", 0)),
                ids=[],
                run_id=result["run_id"],
                dry_run=bool(result["dry_run"]),
                stats=dict(result["stats"]),
                duplicate_relations=list(result.get("duplicate_relations") or []),
            )
        elif request.directory_path:
            result = ingest_directory(
                request.directory_path,
                collection_name=request.collection,
                dry_run=request.dry_run,
                glob=request.glob,
            )
            return KbIngestResponse(
                collection=request.collection or settings.milvus_collection,
                inserted_count=int(result["stats"].get("chunks_upserted", 0)),
                ids=[],
                run_id=result["run_id"],
                dry_run=bool(result["dry_run"]),
                stats=dict(result["stats"]),
                duplicate_relations=list(result.get("duplicate_relations") or []),
            )
        else:
            docs = [item.model_dump() for item in request.docs]

        result = ingest_inline_documents(docs, collection_name=request.collection, dry_run=request.dry_run)
        return KbIngestResponse(
            collection=request.collection or settings.milvus_collection,
            inserted_count=int(result["stats"].get("chunks_upserted", 0)),
            ids=[],
            run_id=result["run_id"],
            dry_run=bool(result["dry_run"]),
            stats=dict(result["stats"]),
            duplicate_relations=list(result.get("duplicate_relations") or []),
        )

    @app.post("/kb/search", response_model=KbSearchResponse)
    def kb_search(request: KbSearchRequest) -> KbSearchResponse:
        metadata_filters = {
            "visibility": request.visibility,
            "biz_domain": request.biz_domain,
            "owner": request.owner,
            "lang": request.lang,
            "source": request.source,
            "source_id": request.source_id,
            "doc_id": request.doc_id,
            "tags": request.tags,
        }
        result = search_knowledge(
            request.query,
            fused_top_k=request.fused_top_k,
            rerank_top_k=request.rerank_top_k,
            metadata_filters=metadata_filters,
        )
        return KbSearchResponse(
            query=result["query"],
            rewritten_queries=list(result["rewritten_queries"]),
            candidate_count=result["candidate_count"],
            reranked_count=result["reranked_count"],
            metadata_filters=dict(result["metadata_filters"]),
            citations=list(result["citations"]),
            results=list(result["results"]),
        )

    @app.post("/memory/write", response_model=MemoryWriteResponse)
    def memory_write(request: MemoryWriteRequest) -> MemoryWriteResponse:
        records = [
            LongTermMemoryRecord(
                memory_id=item.memory_id,
                memory_type=item.memory_type,
                user_id=item.user_id,
                session_id=item.session_id,
                content=item.content,
                tags=list(item.tags),
                metadata=dict(item.metadata),
            )
            for item in request.records
        ]
        written_ids = _memory_store().write(records)
        return MemoryWriteResponse(written_ids=written_ids, count=len(written_ids))

    @app.post("/memory/recall", response_model=MemoryRecallResponse)
    def memory_recall(request: MemoryRecallRequest) -> MemoryRecallResponse:
        records = _memory_store().recall(request.query, user_id=request.user_id, top_k=request.top_k)
        return MemoryRecallResponse(
            query=request.query,
            count=len(records),
            results=[
                {
                    "memory_id": item.memory_id,
                    "memory_type": item.memory_type,
                    "user_id": item.user_id,
                    "session_id": item.session_id,
                    "content": item.content,
                    "tags": list(item.tags),
                    "metadata": dict(item.metadata),
                    "score": item.score,
                    "source": item.source,
                }
                for item in records
            ],
        )

    @app.get("/sessions")
    def list_sessions(limit: int = 50) -> dict[str, Any]:
        items = _list_session_summaries(limit=limit)
        return {"count": len(items), "items": items}

    @app.get("/sessions/{thread_id}")
    def get_session(thread_id: str, limit: int = 20) -> dict[str, Any]:
        graph = build_graph()
        latest_config = _resolve_checkpoint_config(thread_id)
        latest_snapshot = graph.get_state(latest_config)
        checkpoints = [
            _build_checkpoint_history_item(snapshot)
            for snapshot in graph.get_state_history({"configurable": {"thread_id": thread_id}}, limit=limit)
        ]
        return {
            "session": _build_session_summary(latest_snapshot),
            "checkpoints": checkpoints,
            "count": len(checkpoints),
        }

    @app.get("/runs/{thread_id}/state")
    def get_run_state(thread_id: str, checkpoint_id: Optional[str] = None) -> dict[str, Any]:
        snapshot = _get_run_snapshot(thread_id, checkpoint_id=checkpoint_id)
        state = _snapshot_state_values(snapshot)
        return {
            "thread_id": thread_id,
            "checkpoint_id": snapshot.config.get("configurable", {}).get("checkpoint_id"),
            "route": state.get("route"),
            "workflow_status": state.get("workflow_status"),
            "approval_required": bool(state.get("approval_required", False)),
            "approval_status": state.get("approval_status"),
            "next_nodes": list(snapshot.next),
            "pending_interrupts": _extract_snapshot_interrupts(snapshot, state),
            "task_status_suggestion": _build_task_status_suggestion(state),
            "verification_summary": _build_verification_summary(state),
            "state": state,
        }

    @app.get("/skills/catalog")
    def get_skills_catalog() -> dict[str, Any]:
        catalog = build_skill_registry().grouped_specs(
            allowed_business_domains=settings.allowed_skill_business_domains
        )
        return {
            "allowed_business_domains": list(settings.allowed_skill_business_domains),
            "groups": catalog,
            "counts": {
                "all": len(catalog["all"]),
                "regular": len(catalog["regular"]),
                "complex_dev": len(catalog["complex_dev"]),
            },
        }

    @app.get("/metrics/summary")
    def get_metrics_summary() -> dict[str, Any]:
        sessions = _list_session_summaries(limit=None)
        checkpoints = list(build_checkpointer().list(None))
        workflow_counts = Counter(item.get("workflow_status") or "unknown" for item in sessions)
        route_counts = Counter(item.get("route") or "unknown" for item in sessions)
        task_counts = Counter(item["task_status_suggestion"]["code"] for item in sessions)
        verification_counts = Counter(item["verification_summary"]["status"] for item in sessions)
        catalog = build_skill_registry().grouped_specs(
            allowed_business_domains=settings.allowed_skill_business_domains
        )
        return {
            "totals": {
                "sessions": len(sessions),
                "checkpoints": len(checkpoints),
                "skills": len(catalog["all"]),
                "pending_approvals": sum(1 for item in sessions if item.get("approval_status") == "pending"),
                "executed_skill_calls": sum(item["skill_call_counts"]["executed"] for item in sessions),
                "failed_skill_calls": sum(item["skill_call_counts"]["failed"] for item in sessions),
            },
            "workflow_status_counts": dict(workflow_counts),
            "route_counts": dict(route_counts),
            "task_status_suggestion_counts": dict(task_counts),
            "verification_status_counts": dict(verification_counts),
            "last_updated_at": max((item.get("updated_at") or "" for item in sessions), default=""),
        }

    @app.get("/runtime/summary")
    def get_runtime_summary() -> dict[str, Any]:
        sessions = _list_session_summaries(limit=None)
        runtime_events: list[dict[str, Any]] = []
        for item in sessions:
            runtime_events.extend(list(item.get("runtime_events", []) or []))
        summary = build_runtime_summary(runtime_events)
        return {
            "totals": {
                "sessions": len(sessions),
                "runtime_events": summary.get("total_events", 0),
                "total_duration_ms": summary.get("total_duration_ms", 0),
            },
            "status_counts": dict(summary.get("status_counts", {})),
            "validation_status_counts": dict(summary.get("validation_status_counts", {})),
            "skills": dict(summary.get("skills", {})),
        }

    return app


def _build_public_config_payload() -> dict[str, Any]:
    dumped = settings.model_dump()
    sections = {
        section: {field: dumped.get(field) for field in fields}
        for section, fields in PUBLIC_CONFIG_SECTION_FIELDS.items()
    }
    sections["secrets"] = {
        "llm_api_key_configured": _is_secret_configured(settings.llm_api_key, default_placeholder="change-me"),
        "embedding_api_key_configured": _is_secret_configured(settings.embedding_api_key),
        "rerank_api_key_configured": _is_secret_configured(settings.rerank_api_key),
        "milvus_token_configured": _is_secret_configured(settings.milvus_token),
        "prometheus_auth_token_configured": _is_secret_configured(settings.prometheus_auth_token),
        "langsmith_api_key_configured": _is_secret_configured(settings.langsmith_api_key),
    }
    return {
        "sections": sections,
        "editable_fields": list(RUNTIME_EDITABLE_CONFIG_FIELDS),
        "sensitive_fields": sorted(SENSITIVE_CONFIG_FIELDS),
    }


def _is_secret_configured(value: Optional[str], *, default_placeholder: str = "") -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if default_placeholder and text == default_placeholder:
        return False
    return True


def _apply_runtime_config_updates(updates: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if not updates:
        return {}, []

    working_values = settings.model_dump()
    updated: dict[str, Any] = {}
    rejected: list[dict[str, str]] = []

    for field, value in updates.items():
        if field in SENSITIVE_CONFIG_FIELDS:
            rejected.append(
                {
                    "field": field,
                    "reason": "Sensitive settings must be managed via environment variables, SecretRef, or restricted ops channels.",
                }
            )
            continue
        if field not in RUNTIME_EDITABLE_CONFIG_FIELDS:
            rejected.append(
                {
                    "field": field,
                    "reason": "Field is not allowed for runtime updates.",
                }
            )
            continue

        candidate_values = dict(working_values)
        candidate_values[field] = value
        try:
            validated = settings.__class__.model_validate(candidate_values)
        except ValidationError as exc:
            rejected.append({"field": field, "reason": _validation_error_message(exc, field)})
            continue

        working_values = validated.model_dump()
        updated[field] = working_values[field]

    for field, value in updated.items():
        setattr(settings, field, value)

    if updated:
        build_graph.cache_clear()
        build_checkpointer.cache_clear()
        _memory_store.cache_clear()

    return updated, rejected


def _validation_error_message(exc: ValidationError, field: str) -> str:
    for item in exc.errors():
        loc = item.get("loc", ())
        if loc and loc[0] == field:
            return item.get("msg", "Validation failed.")
    if exc.errors():
        return exc.errors()[0].get("msg", "Validation failed.")
    return "Validation failed."


def _resolve_checkpoint_config(thread_id: str, checkpoint_id: Optional[str] = None) -> dict[str, Any]:
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if checkpoint_id:
        config["configurable"]["checkpoint_id"] = checkpoint_id
    checkpoint_tuple = build_checkpointer().get_tuple(config)
    if checkpoint_tuple is None:
        raise HTTPException(status_code=404, detail=f"Unknown thread_id: {thread_id}")
    return checkpoint_tuple.config


def _get_run_snapshot(thread_id: str, checkpoint_id: Optional[str] = None) -> Any:
    config = _resolve_checkpoint_config(thread_id, checkpoint_id=checkpoint_id)
    return build_graph().get_state(config)


def _list_session_summaries(limit: Optional[int]) -> list[dict[str, Any]]:
    latest_configs: dict[str, dict[str, Any]] = {}
    for checkpoint in build_checkpointer().list(None):
        config = checkpoint.config
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id and thread_id not in latest_configs:
            latest_configs[thread_id] = config

    graph = build_graph()
    items = [_build_session_summary(graph.get_state(config)) for config in latest_configs.values()]
    items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    if limit is None:
        return items
    return items[:limit]


def _build_session_summary(snapshot: Any) -> dict[str, Any]:
    state = _snapshot_state_values(snapshot)
    executed_skill_calls = list(state.get("executed_skill_calls", []) or [])
    runtime_events = list(state.get("runtime_events", []) or [])
    return {
        "thread_id": snapshot.config.get("configurable", {}).get("thread_id"),
        "checkpoint_id": snapshot.config.get("configurable", {}).get("checkpoint_id"),
        "route": state.get("route"),
        "workflow_status": state.get("workflow_status"),
        "approval_required": bool(state.get("approval_required", False)),
        "approval_status": state.get("approval_status"),
        "updated_at": snapshot.created_at,
        "last_user_query": state.get("user_query", ""),
        "final_answer": _extract_final_answer(state),
        "pending_interrupts": _extract_snapshot_interrupts(snapshot, state),
        "task_status_suggestion": _build_task_status_suggestion(state),
        "verification_summary": _build_verification_summary(state),
        "skill_call_counts": {
            "planned": len(list(state.get("planned_skill_calls", []) or [])),
            "executed": len(executed_skill_calls),
            "failed": sum(1 for item in executed_skill_calls if str(item.get("status", "")).strip() == "failed"),
        },
        "runtime_events": runtime_events,
        "runtime_summary": dict(state.get("runtime_summary", {}) or {}),
    }


def _build_checkpoint_history_item(snapshot: Any) -> dict[str, Any]:
    state = _snapshot_state_values(snapshot)
    parent_config = snapshot.parent_config or {}
    return {
        "thread_id": snapshot.config.get("configurable", {}).get("thread_id"),
        "checkpoint_id": snapshot.config.get("configurable", {}).get("checkpoint_id"),
        "parent_checkpoint_id": parent_config.get("configurable", {}).get("checkpoint_id"),
        "created_at": snapshot.created_at,
        "route": state.get("route"),
        "workflow_status": state.get("workflow_status"),
        "approval_status": state.get("approval_status"),
        "next_nodes": list(snapshot.next),
        "task_status_suggestion": _build_task_status_suggestion(state),
        "verification_summary": _build_verification_summary(state),
    }


def _snapshot_state_values(snapshot: Any) -> dict[str, Any]:
    values = getattr(snapshot, "values", {}) or {}
    if isinstance(values, dict):
        return values
    return {"value": values}


def _build_token_usage(snapshot_values: dict[str, Any]) -> dict[str, Any]:
    """基于 state.messages 实时估算当前 thread 的上下文用量，返回 JSON 友好结构。"""

    try:
        from ops_rag_agent.memory.tokens import build_usage_snapshot
    except Exception:
        return dict(snapshot_values.get("token_usage", {}) or {})

    snapshot = build_usage_snapshot(
        messages=list(snapshot_values.get("messages", []) or []),
        rolling_summary=str(snapshot_values.get("rolling_summary", "") or ""),
        context_window=settings.llm_context_window_tokens,
        encoding_name=settings.tiktoken_encoding_name,
    )
    # 合并 checkpointer 里已有的 token_usage（保留 last_prompt_tokens 等字段）
    stored = dict(snapshot_values.get("token_usage", {}) or {})
    stored.update(snapshot)
    return stored


def _extract_snapshot_interrupts(snapshot: Any, state: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in getattr(snapshot, "interrupts", ()) or ():
        items.append({"interrupt_id": getattr(item, "id", ""), "value": getattr(item, "value", None)})

    for item in state.get("pending_interrupts", []) or []:
        if item not in items:
            items.append(item)
    return items


def _build_task_status_suggestion(state: dict[str, Any]) -> dict[str, Any]:
    workflow_status = str(state.get("workflow_status", "")).strip().lower()
    approval_status = str(state.get("approval_status", "")).strip().lower()
    actions = list(state.get("remediation_actions", []) or [])
    action_statuses = [str(item.get("status", "planned")).strip().lower() for item in actions]
    completed_actions = sum(1 for status in action_statuses if status == "done")
    failed_actions = sum(1 for status in action_statuses if status == "failed")
    open_actions = sum(1 for status in action_statuses if status not in {"done", "failed", "skipped"})
    verification_summary = _build_verification_summary(state)

    if workflow_status == "rejected":
        code = "stopped"
        summary = "流程已停止，建议人工复核后重新发起。"
    elif approval_status == "pending" or workflow_status == "waiting_approval":
        code = "awaiting_approval"
        summary = "存在待审批动作，建议先完成审批。"
    elif failed_actions > 0:
        code = "needs_attention"
        summary = "存在执行失败动作，建议先排查失败原因。"
    elif open_actions > 0:
        code = "ready_to_execute"
        summary = "存在未完成动作，建议继续执行或补充决策。"
    elif state.get("execution_results") and verification_summary["status"] in {"pending", "not_run"}:
        code = "verify_changes"
        summary = "已产生执行结果，建议补充验证结果。"
    elif workflow_status == "completed":
        code = "completed"
        summary = "流程已完成。"
    elif workflow_status == "running":
        code = "in_progress"
        summary = "流程仍在运行，建议继续观察。"
    else:
        code = "review_state"
        summary = "建议结合当前状态做人工复核。"

    return {
        "code": code,
        "summary": summary,
        "open_action_count": open_actions,
        "completed_action_count": completed_actions,
        "failed_action_count": failed_actions,
    }


def _build_verification_summary(state: dict[str, Any]) -> dict[str, Any]:
    results = [str(item).strip() for item in state.get("verification_results", []) or [] if str(item).strip()]
    if not results:
        if state.get("execution_results"):
            return {
                "status": "pending",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pending": 0,
                "latest_result": "",
                "summary": "已有执行结果，待补充验证结果。",
            }
        return {
            "status": "not_run",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pending": 0,
            "latest_result": "",
            "summary": "暂无验证结果。",
        }

    counts = Counter(_classify_verification_result(item) for item in results)
    if counts["failed"] > 0:
        status = "failed"
        summary = "存在失败验证结果。"
    elif counts["pending"] > 0 and counts["passed"] > 0:
        status = "mixed"
        summary = "验证结果部分通过，仍需补充确认。"
    elif counts["pending"] > 0:
        status = "pending"
        summary = "验证结果未明确，需要进一步确认。"
    else:
        status = "passed"
        summary = "验证结果通过。"

    return {
        "status": status,
        "total": len(results),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "pending": counts["pending"],
        "latest_result": results[-1],
        "summary": summary,
    }


def _classify_verification_result(text: str) -> str:
    lowered = text.lower()
    passed_patterns = (
        r"\bpass(?:ed)?\b",
        r"\bsuccess(?:ful)?\b",
        r"\bok\b",
        r"\bhealthy\b",
        r"\bverified\b",
    )
    failed_patterns = (
        r"\bfail(?:ed)?\b",
        r"\berror\b",
        r"\btimeout\b",
        r"\bunhealthy\b",
        r"\breject(?:ed)?\b",
    )
    if any(re.search(pattern, lowered) for pattern in failed_patterns) or any(
        token in text for token in ("失败", "错误", "异常", "超时", "未恢复", "拒绝")
    ):
        return "failed"
    if any(re.search(pattern, lowered) for pattern in passed_patterns) or any(
        token in text for token in ("通过", "成功", "正常", "健康", "已恢复", "验证通过")
    ):
        return "passed"
    return "pending"


def _extract_final_answer(result: dict[str, Any]) -> str:
    if result.get("__interrupt__"):
        return "流程已暂停，等待审批后继续。"

    final_answer = result.get("final_answer")
    if isinstance(final_answer, str) and final_answer:
        return final_answer

    messages = result.get("messages", [])
    if not messages:
        return "流程已结束。"

    last_message = messages[-1]
    content = getattr(last_message, "content", last_message)
    if isinstance(content, str):
        return content
    return str(content)


def _extract_interrupts(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw_interrupts = result.get("__interrupt__", [])
    items: list[dict[str, Any]] = []
    for item in raw_interrupts:
        items.append(
            {
                "interrupt_id": getattr(item, "id", ""),
                "value": getattr(item, "value", None),
            }
        )
    return items


app = create_app()
