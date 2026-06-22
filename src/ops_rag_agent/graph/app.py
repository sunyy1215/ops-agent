from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ops_rag_agent.agents.intent_analyzer import (
    IntentAnalysis,
    analyze_intent,
    heuristic_intent,
)
from ops_rag_agent.agents.planner import (
    heuristic_plan,
    plan_execution,
)
from ops_rag_agent.agents.skill_router_node import run_skill_router_node
from ops_rag_agent.config import settings
from ops_rag_agent.guardrails import allow_guardrail_result, merge_guardrail_state, review_input_guardrails
from ops_rag_agent.graph.checkpointing import build_checkpointer
from ops_rag_agent.memory.compression import (
    recompress_with_reflection,
    reflect_compression_summary,
    should_compress,
    split_for_compression,
    summarize_middle_zone,
)
from ops_rag_agent.memory.store import (
    InMemoryLongTermMemoryStore,
    LongTermMemoryRecord,
    MilvusLongTermMemoryStore,
)
from ops_rag_agent.models.factory import build_chat_llm
from ops_rag_agent.observability import append_audit_event, trace_graph_node
from ops_rag_agent.prompts import get_prompt_spec, load_prompt_text
from ops_rag_agent.schemas.state import AgentState
from ops_rag_agent.skills.bootstrap import build_skill_registry


@lru_cache(maxsize=1)
def build_graph() -> Any:
    registry = build_skill_registry()
    memory_store = (
        MilvusLongTermMemoryStore()
        if settings.long_term_memory_backend == "milvus"
        else InMemoryLongTermMemoryStore()
    )

    graph = StateGraph(AgentState)
    graph.add_node(
        "prepare_input",
        trace_graph_node("prepare_input")(lambda state: prepare_input(state, memory_store)),
    )
    graph.add_node(
        "analyze_intent",
        trace_graph_node("analyze_intent")(analyze_intent_node),
    )
    graph.add_node(
        "plan",
        trace_graph_node("plan")(lambda state: plan_node(state, registry)),
    )
    graph.add_node(
        "skill_router",
        trace_graph_node("skill_router")(lambda state: run_skill_router_node(state, registry)),
    )
    graph.add_node("approval_gate", trace_graph_node("approval_gate")(approval_gate))
    graph.add_node("memory_compressor", trace_graph_node("memory_compressor")(memory_compressor))
    graph.add_node("memory_reflection", trace_graph_node("memory_reflection")(memory_reflection))
    graph.add_node(
        "finalize",
        trace_graph_node("finalize")(lambda state: finalize(state, memory_store)),
    )

    # 新流水线：prepare -> analyze_intent -> plan -> skill_router(execute+conclude)
    #          -> (approval?) -> memory_compressor -> memory_reflection -> finalize
    graph.add_edge(START, "prepare_input")
    graph.add_conditional_edges(
        "prepare_input",
        after_prepare_input,
        {
            "analyze_intent": "analyze_intent",
            "finalize": "finalize",
        },
    )
    graph.add_edge("analyze_intent", "plan")
    graph.add_conditional_edges(
        "plan",
        after_plan,
        {
            "skill_router": "skill_router",
            "memory_compressor": "memory_compressor",
        },
    )
    graph.add_conditional_edges(
        "skill_router",
        after_skill_router,
        {
            "approval_gate": "approval_gate",
            "memory_compressor": "memory_compressor",
        },
    )
    graph.add_conditional_edges(
        "approval_gate",
        after_approval_gate,
        {
            "skill_router": "skill_router",
            "memory_compressor": "memory_compressor",
        },
    )
    graph.add_edge("memory_compressor", "memory_reflection")
    graph.add_conditional_edges(
        "memory_reflection",
        after_memory_reflection,
        {
            "memory_compressor": "memory_compressor",
            "finalize": "finalize",
        },
    )
    graph.add_edge("finalize", END)
    return graph.compile(
        checkpointer=build_checkpointer(),
        name=settings.langgraph_graph_name,
    )


def invoke_graph(state: dict[str, Any], *, thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
    """Convenience wrapper for invoking the compiled graph with required checkpointer config."""

    graph = build_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if checkpoint_id:
        config["configurable"]["checkpoint_id"] = checkpoint_id
    return graph.invoke(state, config=config)


def prepare_input(state: AgentState, memory_store: InMemoryLongTermMemoryStore) -> dict[str, Any]:
    query = state.get("user_query", "")
    prompt_spec = get_prompt_spec("supervisor.prepare_input")
    system_prompt = load_prompt_text("supervisor.prepare_input")
    memory_hits = memory_store.recall(
        query,
        user_id=state.get("user_id"),
        top_k=settings.long_term_memory_top_k,
    )
    short_term_memory = {
        "active_summary": state.get("rolling_summary", ""),
        "protected_head_count": len(state.get("protected_head", [])),
        "protected_tail_count": len(state.get("protected_tail", [])),
        "last_compression_status": "idle",
        "last_token_usage": state.get("token_usage", {}).get("total", 0),
    }

    if not state.get("messages"):
        guardrail_result = (
            review_input_guardrails(
                query,
                enable_model_review=settings.guardrails_model_review_enabled,
            )
            if settings.guardrails_enabled
            else allow_guardrail_result()
        )
        updates = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query),
            ],
            "workflow_status": "running",
            "approval_status": state.get("approval_status", "not_required"),
            "short_term_memory": short_term_memory,
            "long_term_memory_hits": [_memory_record_to_dict(item) for item in memory_hits],
            "prompt_versions": {
                **state.get("prompt_versions", {}),
                "prepare_input": prompt_spec.version,
            },
        }
        updates.update(
            merge_guardrail_state(
                state,
                guardrail_result,
                block_message="The request was blocked by input guardrails.",
            )
        )
        if updates.get("guardrail_action") == "block":
            updates["route"] = "end"
        updates["audit_trail"] = append_audit_event(
            state,
            "graph_started",
            node="prepare_input",
            details={
                "query_length": len(query),
                "thread_id": state.get("thread_id", ""),
                "prompt_version": prompt_spec.version,
                "guardrail_action": updates.get("guardrail_action", "allow"),
            },
        )
        return updates

    # 多轮场景：同一 thread 接收到新一轮 user_query
    # 需要把新问题作为 HumanMessage 追加，并清空上一轮 turn-scope 残留状态，
    # 否则 after_plan 等节点会读到上一轮的 final_answer/router_scratchpad，
    # 直接走 memory_compressor 返回上一轮结果，造成"卡在上轮输入"的现象。
    last_human_content = ""
    for prev_msg in reversed(state.get("messages") or []):
        if isinstance(prev_msg, HumanMessage):
            last_human_content = str(getattr(prev_msg, "content", "") or "")
            break

    if query and query != last_human_content:
        guardrail_result = (
            review_input_guardrails(
                query,
                enable_model_review=settings.guardrails_model_review_enabled,
            )
            if settings.guardrails_enabled
            else allow_guardrail_result()
        )
        updates = {
            # add_messages reducer 会把新 HumanMessage 追加到既有 messages
            "messages": [HumanMessage(content=query)],
            "workflow_status": "running",
            "approval_status": "not_required",
            "approval_required": False,
            "short_term_memory": short_term_memory,
            "long_term_memory_hits": [_memory_record_to_dict(item) for item in memory_hits],
            # 清空上一轮 turn-scope 状态
            "final_answer": "",
            "router_scratchpad": [],
            "router_iterations": 0,
            "router_stop_reason": "",
            "router_evidence": {},
            "router_validation_results": [],
            "router_runtime_observations": [],
            "runtime_audit_records": [],
            "runtime_events": [],
            "runtime_summary": {},
            "intent_analysis": {},
            "execution_plan": {},
            "plan_progress": [],
            "citations": [],
            "route": "",
        }
        updates.update(
            merge_guardrail_state(
                state,
                guardrail_result,
                block_message="The request was blocked by input guardrails.",
            )
        )
        if updates.get("guardrail_action") == "block":
            updates["route"] = "end"
        updates["audit_trail"] = append_audit_event(
            state,
            "new_turn_started",
            node="prepare_input",
            details={
                "query_length": len(query),
                "thread_id": state.get("thread_id", ""),
                "guardrail_action": updates.get("guardrail_action", "allow"),
            },
        )
        return updates

    return {
        "workflow_status": state.get("workflow_status", "running"),
        "short_term_memory": short_term_memory,
        "long_term_memory_hits": [_memory_record_to_dict(item) for item in memory_hits],
    }


def after_prepare_input(state: AgentState) -> str:
    """prepare_input 结束后的分流：guardrail 拦截直接收尾，否则进意图分析。"""
    if state.get("route") == "end" or state.get("workflow_status") == "rejected":
        return "finalize"
    return "analyze_intent"


def _default_llm_invoke(prompt: str) -> str:
    """默认 LLM 调用器（供 intent / plan 节点共用）。"""

    llm = build_chat_llm()
    msg = llm.invoke(prompt)
    return getattr(msg, "content", str(msg))


def analyze_intent_node(state: AgentState) -> dict[str, Any]:
    """节点：先做用户意图分析，写入 state.intent_analysis。"""

    user_query = str(state.get("user_query") or "").strip()
    try:
        intent = analyze_intent(user_query=user_query, llm_invoke=_default_llm_invoke)
        source = "llm"
    except Exception as exc:  # noqa: BLE001
        intent = heuristic_intent(user_query)
        intent.reasoning = f"{intent.reasoning or 'heuristic_fallback'} (llm_error: {type(exc).__name__})"
        source = "heuristic"

    return {
        "intent_analysis": intent.to_dict(),
        "audit_trail": append_audit_event(
            state,
            "intent_analyzed",
            node="analyze_intent",
            details={
                "source": source,
                "complexity": intent.complexity,
                "need_tools": intent.need_tools,
                "domain_hints": list(intent.domain_hints),
                "sub_questions": len(intent.sub_questions),
            },
        ),
    }


def plan_node(state: AgentState, registry: Any) -> dict[str, Any]:
    """节点：基于意图分析生成执行计划，写入 state.execution_plan。"""

    raw_intent = state.get("intent_analysis") or {}
    intent = IntentAnalysis(
        summary=str(raw_intent.get("summary") or ""),
        sub_questions=list(raw_intent.get("sub_questions") or []),
        domain_hints=list(raw_intent.get("domain_hints") or []),
        complexity=str(raw_intent.get("complexity") or "moderate"),
        need_tools=bool(raw_intent.get("need_tools", True)),
        reasoning=str(raw_intent.get("reasoning") or ""),
    )

    allowed = (
        list(settings.allowed_skill_business_domains)
        if settings.allowed_skill_business_domains
        else None
    )

    try:
        plan = plan_execution(
            intent=intent,
            registry=registry,
            llm_invoke=_default_llm_invoke,
            allowed_business_domains=allowed,
        )
        source = "llm"
    except Exception as exc:  # noqa: BLE001
        plan = heuristic_plan(intent)
        plan.reasoning = f"{plan.reasoning or 'heuristic_fallback'} (llm_error: {type(exc).__name__})"
        source = "heuristic"

    return {
        "execution_plan": plan.to_dict(),
        "plan_progress": [],
        "audit_trail": append_audit_event(
            state,
            "plan_generated",
            node="plan",
            details={
                "source": source,
                "step_count": len(plan.steps),
                "skills": [s.suggested_skill_id for s in plan.steps if s.suggested_skill_id],
            },
        ),
    }


def after_plan(_state: AgentState) -> str:
    """plan 节点之后的分流：始终进 skill_router 执行（plan 至少包含一步 rag.search）。"""

    return "skill_router"


def after_skill_router(state: AgentState) -> str:
    if state.get("approval_required"):
        return "approval_gate"
    return "memory_compressor"


def approval_gate(state: AgentState) -> dict[str, Any]:
    payload = state.get("approval_payload", {})
    decision = interrupt(
        {
            "type": "approval_required",
            "resumable_from": "approval_gate",
            "payload": payload,
        }
    )

    approval_status = _normalize_approval_status(decision)
    approved = approval_status == "approved"
    msg = (
        "审批已通过，正在继续执行后续处置流程。"
        if approved
        else "审批未通过，已跳过该高危 skill；router 可继续尝试其他方案。"
    )
    # 新架构下所有审批都来自 skill_router 的 skill_invocation；被拒也不终止工作流
    updates: dict[str, Any] = {
        "approval_status": approval_status,
        "workflow_status": "running",
        "resumable_from": "",
        "messages": [AIMessage(content=msg)],
        "audit_trail": append_audit_event(
            state,
            "approval_resolved",
            node="approval_gate",
            details={"approval_status": approval_status},
        ),
    }
    return updates


def after_approval_gate(_state: AgentState) -> str:
    # 审批节点之后，统一回到 skill_router 让 LLM 继续推理下一步
    return "skill_router"


def memory_compressor(state: AgentState) -> dict[str, Any]:
    messages = _messages_to_dicts(state.get("messages", []))
    # 用真实 tokenizer 统计 + 暴露给前端的 token_usage 快照
    from ops_rag_agent.memory.tokens import build_usage_snapshot

    usage_snapshot = build_usage_snapshot(
        messages=messages,
        rolling_summary=state.get("rolling_summary", ""),
        context_window=settings.llm_context_window_tokens,
        encoding_name=settings.tiktoken_encoding_name,
    )
    token_usage = {"total": usage_snapshot["total"], **usage_snapshot}

    if not should_compress(
        token_usage,
        max_context_tokens=settings.llm_context_window_tokens,
        trigger_ratio=settings.compression_trigger_ratio,
    ):
        return {
            "token_usage": token_usage,
            "short_term_memory": {
                "active_summary": state.get("rolling_summary", ""),
                "protected_head_count": len(state.get("protected_head", [])),
                "protected_tail_count": len(state.get("protected_tail", [])),
                "last_compression_status": "skipped",
                "last_token_usage": token_usage.get("total", 0),
            },
            "compression_retry_requested": False,
            "compression_reflection": {
                "status": "skipped",
                "passed": False,
                "feedback": ["compression threshold not reached"],
                "missing_sections": [],
            },
            "audit_trail": append_audit_event(
                state,
                "compression_skipped",
                node="memory_compressor",
                details={"token_total": token_usage.get("total", 0)},
            ),
        }

    head, middle, tail = split_for_compression(messages)
    if not middle:
        return {
            "token_usage": token_usage,
            "protected_head": head,
            "compression_zone": middle,
            "protected_tail": tail,
            "rolling_summary": "",
            "compression_retry_requested": False,
            "compression_reflection": {
                "status": "skipped",
                "passed": False,
                "feedback": ["no middle zone to compress"],
                "missing_sections": [],
            },
            "short_term_memory": {
                "active_summary": "",
                "protected_head_count": len(head),
                "protected_tail_count": len(tail),
                "last_compression_status": "skipped",
                "last_token_usage": token_usage.get("total", 0),
            },
            "audit_trail": append_audit_event(
                state,
                "compression_skipped",
                node="memory_compressor",
                details={"reason": "no_middle_zone"},
            ),
        }

    retry_count = state.get("compression_retry_count", 0)
    reflection_feedback = state.get("compression_reflection", {}).get("feedback", [])
    if retry_count > 0 and reflection_feedback:
        summary = recompress_with_reflection(middle, reflection_feedback, retry_count)
        compression_status = "retried"
    else:
        summary = summarize_middle_zone(middle)
        compression_status = "compressed"

    return {
        "token_usage": token_usage,
        "protected_head": head,
        "compression_zone": middle,
        "protected_tail": tail,
        "rolling_summary": summary,
        "compression_retry_requested": False,
        "short_term_memory": {
            "active_summary": summary,
            "protected_head_count": len(head),
            "protected_tail_count": len(tail),
            "last_compression_status": compression_status,
            "last_token_usage": token_usage.get("total", 0),
        },
        "audit_trail": append_audit_event(
            state,
            "compression_completed",
            node="memory_compressor",
            details={
                "status": compression_status,
                "middle_message_count": len(middle),
                "token_total": token_usage.get("total", 0),
            },
        ),
    }


def memory_reflection(state: AgentState) -> dict[str, Any]:
    summary = state.get("rolling_summary", "")
    middle = state.get("compression_zone", [])
    if not summary:
        return {
            "compression_reflection": state.get(
                "compression_reflection",
                {
                    "status": "skipped",
                    "passed": False,
                    "feedback": ["summary is empty"],
                    "missing_sections": ["task_goal", "confirmed_facts", "actions_taken"],
                },
            ),
            "compression_retry_requested": False,
            "audit_trail": append_audit_event(
                state,
                "compression_reflection_skipped",
                node="memory_reflection",
                details={"reason": "summary_empty"},
            ),
        }

    reflection = reflect_compression_summary(summary, middle)

    updates: dict[str, Any] = {"compression_reflection": reflection}
    if reflection.get("passed"):
        updates["audit_trail"] = append_audit_event(
            state,
            "compression_reflection_passed",
            node="memory_reflection",
            details={"feedback": list(reflection.get("feedback", []))},
        )
        return updates

    retry_count = state.get("compression_retry_count", 0)
    if summary and retry_count < settings.max_compression_retries:
        updates["compression_retry_count"] = retry_count + 1
        updates["compression_retry_requested"] = True
    else:
        updates["compression_retry_requested"] = False
    updates["audit_trail"] = append_audit_event(
        state,
        "compression_reflection_failed",
        node="memory_reflection",
        details={
            "retry_requested": updates["compression_retry_requested"],
            "missing_sections": list(reflection.get("missing_sections", [])),
        },
    )
    return updates


def after_memory_reflection(state: AgentState) -> str:
    reflection = state.get("compression_reflection", {})
    if not state.get("rolling_summary"):
        return "finalize"
    if reflection.get("passed"):
        return "finalize"
    if state.get("compression_retry_requested"):
        return "memory_compressor"
    return "finalize"


def finalize(state: AgentState, memory_store: InMemoryLongTermMemoryStore) -> dict[str, Any]:
    final_answer = state.get("final_answer", "Workflow finished.")
    queued_records = _build_long_term_memory_writes(state)
    written_ids: list[str] = []
    if queued_records:
        written_ids = memory_store.write(queued_records)

    return {
        "messages": [AIMessage(content=final_answer)],
        "long_term_memory_write_queue": [_memory_record_to_dict(item) for item in queued_records],
        "long_term_memory_written_ids": written_ids,
        "workflow_status": state.get("workflow_status", "completed")
        if state.get("workflow_status") == "rejected"
        else "completed",
        "audit_trail": append_audit_event(
            state,
            "graph_finalized",
            node="finalize",
            details={"memory_write_count": len(written_ids)},
        ),
    }


def _messages_to_dicts(messages: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))
        items.append({"role": role, "content": content})
    return items


def _normalize_approval_status(decision: Any) -> str:
    if isinstance(decision, bool):
        return "approved" if decision else "rejected"
    if isinstance(decision, str):
        normalized = decision.strip().lower()
        if normalized in {"approved", "approve", "yes", "y", "true"}:
            return "approved"
        if normalized in {"rejected", "reject", "no", "n", "false"}:
            return "rejected"
    if isinstance(decision, dict):
        for key in ("status", "decision", "approved"):
            if key in decision:
                return _normalize_approval_status(decision[key])
    return "rejected"


def _build_long_term_memory_writes(state: AgentState) -> list[LongTermMemoryRecord]:
    query = state.get("user_query", "").strip()
    if not query:
        return []

    final_answer = state.get("final_answer", "").strip()
    if not final_answer:
        return []

    # 新架构里没有 dialog/ops/rag 这种大类标签，统一用 router_stop_reason / 调用过的 skill 列表来打 tag
    skills_used = sorted(
        {
            str(entry.get("skill_id"))
            for entry in (state.get("router_scratchpad") or [])
            if entry.get("skill_id")
        }
    )
    has_ops_skill = any(s.startswith("ops.") for s in skills_used)
    memory_type = "incident" if has_ops_skill else "knowledge_summary"
    tags = ["skill_router"]
    if skills_used:
        tags.append("skills:" + ",".join(skills_used)[:200])

    record = LongTermMemoryRecord(
        memory_id=f"memory-{abs(hash((query, final_answer))) % 10_000_000}",
        memory_type=memory_type,
        user_id=state.get("user_id"),
        session_id=state.get("session_id"),
        content=f"user_query={query}\nfinal_answer={final_answer}",
        tags=tags,
        metadata={
            "citations": state.get("citations", []),
            "summary": state.get("rolling_summary", ""),
            "skills_used": skills_used,
            "router_iterations": state.get("router_iterations", 0),
            "router_stop_reason": state.get("router_stop_reason", ""),
        },
        source="memory://graph-finalize",
    )
    return [record]


def _memory_record_to_dict(record: LongTermMemoryRecord) -> dict[str, Any]:
    return {
        "memory_id": record.memory_id,
        "memory_type": record.memory_type,
        "user_id": record.user_id,
        "session_id": record.session_id,
        "content": record.content,
        "tags": list(record.tags),
        "metadata": dict(record.metadata),
        "score": record.score,
        "source": record.source,
    }
