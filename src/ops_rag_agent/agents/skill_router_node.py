"""LangGraph 节点：把 `run_skill_router` 适配成一个 graph 节点。

职责：
  - 从 AgentState 里抽取 user_query / 上下文，调用纯函数版 `run_skill_router`
  - 把 `RouterResult` 回写进 AgentState：
      - final_answer / workflow_status / audit_trail
      - router_scratchpad / router_iterations / router_stop_reason
      - ops_evidence（复用已有字段展示证据）
  - 如果 `pending_approval` 非空，设置 `approval_required=True` + `approval_payload`，
    由 `approval_gate` 节点统一走 `interrupt`。审批恢复时再由下一次图调度把
    `approval_resume={...}` 翻译出来喂回来。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from ops_rag_agent.agents.skill_router import (
    DEFAULT_MAX_ITERATIONS,
    RouterResult,
    ScratchpadEntry,
    build_native_router_tools,
    run_skill_router,
)
from ops_rag_agent.config import settings
from ops_rag_agent.models.factory import build_router_llm
from ops_rag_agent.observability import append_audit_event
from ops_rag_agent.skills.registry import SkillRegistry
from ops_rag_agent.skills.runtime import build_runtime_event, build_runtime_summary, runtime_result_to_skill_call


def _default_llm_invoke(prompt: str) -> Any:
    tools = build_native_router_tools() if settings.router_native_tool_calling_enabled else None
    llm = build_router_llm(native_tools=tools)
    return llm.invoke(prompt)


def _scratchpad_to_dicts(scratchpad: list[ScratchpadEntry]) -> list[dict[str, Any]]:
    return [
        {
            "turn": e.turn,
            "thought": e.thought,
            "action": e.action,
            "skill_id": e.skill_id,
            "arguments": e.arguments,
            "observation": e.observation,
            "status": e.status,
            "plan_step_id": e.plan_step_id,
            "validation": dict(e.validation or {}),
            "observation_data": dict(e.observation_data or {}),
            "runtime_audit": list(e.runtime_audit or []),
            "decision_source": e.decision_source,
        }
        for e in scratchpad
    ]


def _dicts_to_scratchpad(items: list[dict[str, Any]]) -> list[ScratchpadEntry]:
    out: list[ScratchpadEntry] = []
    for item in items or []:
        out.append(
            ScratchpadEntry(
                turn=int(item.get("turn") or len(out) + 1),
                thought=str(item.get("thought") or ""),
                action=str(item.get("action") or "call_skill"),
                skill_id=str(item.get("skill_id") or ""),
                arguments=dict(item.get("arguments") or {}),
                observation=str(item.get("observation") or ""),
                status=str(item.get("status") or "done"),
                plan_step_id=str(item.get("plan_step_id") or ""),
                validation=dict(item.get("validation") or {}),
                observation_data=dict(item.get("observation_data") or {}),
                runtime_audit=list(item.get("runtime_audit") or []),
                decision_source=str(item.get("decision_source") or ""),
            )
        )
    return out


def _build_approval_payload(pending: Any) -> dict[str, Any]:
    return {
        "type": "skill_invocation",
        "skill_id": pending.skill_id,
        "arguments": pending.arguments,
        "risk": pending.risk_level,
        "reason": pending.reason,
        "thought": pending.thought,
    }


def _extract_approval_resume(state: dict) -> dict[str, Any] | None:
    """审批通过后，上游 approval_gate 会把 approval_status=approved 写回状态，
    这里再看 approval_payload 的 type 是否为 skill_invocation，只有这样才是
    我们要 resume 的 skill 调用。
    """

    status = str(state.get("approval_status") or "")
    if status not in {"approved", "rejected"}:
        return None
    payload = state.get("approval_payload") or {}
    if str(payload.get("type") or "") != "skill_invocation":
        return None
    return {
        "approved": status == "approved",
        "skill_id": str(payload.get("skill_id") or ""),
        "arguments": dict(payload.get("arguments") or {}),
    }


def _skill_manifest_for_event(registry: SkillRegistry, skill_id: str) -> dict[str, Any]:
    if not skill_id:
        return {}
    try:
        return registry.get(skill_id).spec.to_manifest()
    except KeyError:
        return {}


def _compute_plan_progress(
    execution_plan: dict[str, Any],
    scratchpad: list[ScratchpadEntry],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """基于 scratchpad 更新 plan.steps 的 status 并生成 plan_progress。

    返回 (updated_execution_plan, plan_progress)
    """

    plan = dict(execution_plan or {})
    steps = [dict(s) for s in (plan.get("steps") or [])]

    # 建立 step_id / skill_id → step 索引的多路映射
    step_by_id = {str(s.get("id") or ""): s for s in steps if s.get("id")}
    step_by_skill: dict[str, list[dict[str, Any]]] = {}
    for s in steps:
        sid = str(s.get("suggested_skill_id") or "")
        if sid:
            step_by_skill.setdefault(sid, []).append(s)

    progress: list[dict[str, Any]] = []
    # 标记哪些 skill_id 已经被消耗过（FIFO 匹配计划里多次使用同一 skill 的情况）
    consumed_step_ids: set[str] = set()

    for entry in scratchpad:
        if entry.action != "call_skill":
            continue
        matched: dict[str, Any] | None = None
        # 1. LLM 明确声明 plan_step_id
        if entry.plan_step_id and entry.plan_step_id in step_by_id:
            matched = step_by_id[entry.plan_step_id]
        # 2. 按 skill_id 匹配第一个未消费的计划步骤
        if matched is None:
            for candidate in step_by_skill.get(entry.skill_id, []):
                if str(candidate.get("id")) not in consumed_step_ids:
                    matched = candidate
                    break

        if matched is not None:
            sid = str(matched.get("id") or "")
            if sid:
                consumed_step_ids.add(sid)
            if entry.status == "done":
                matched["status"] = "done"
            elif entry.status == "failed":
                matched["status"] = "failed"
            elif entry.status == "pending_approval":
                matched["status"] = "pending_approval"
            else:
                matched["status"] = entry.status or "running"

        progress.append(
            {
                "step_id": str(matched.get("id")) if matched else "",
                "turn": entry.turn,
                "skill_id": entry.skill_id,
                "status": entry.status,
                "observation": (entry.observation or "")[:1200],
                "thought": entry.thought,
                "arguments": dict(entry.arguments or {}),
            }
        )

    # 所有 scratchpad 遍历完后，把剩余未命中的 planned 步骤维持 planned
    plan["steps"] = steps
    return plan, progress


def run_skill_router_node(
    state: dict,
    registry: SkillRegistry,
    *,
    llm_invoke=None,
) -> dict[str, Any]:
    """Graph 节点入口。"""

    user_query = str(state.get("user_query") or "")
    invoke = llm_invoke or _default_llm_invoke

    initial_scratchpad = _dicts_to_scratchpad(state.get("router_scratchpad") or [])
    initial_evidence: dict[str, Any] = dict(state.get("router_evidence") or {})
    approval_resume = _extract_approval_resume(state)

    context_text = state.get("rolling_summary") or ""
    intent_analysis = dict(state.get("intent_analysis") or {})
    execution_plan = dict(state.get("execution_plan") or {})

    result: RouterResult = run_skill_router(
        user_query=user_query,
        registry=registry,
        llm_invoke=invoke,
        allowed_business_domains=list(settings.allowed_skill_business_domains)
        if settings.allowed_skill_business_domains
        else None,
        max_iterations=DEFAULT_MAX_ITERATIONS,
        initial_evidence=initial_evidence,
        initial_scratchpad=initial_scratchpad,
        approval_resume=approval_resume,
        context_text=context_text,
        intent_analysis=intent_analysis or None,
        execution_plan=execution_plan or None,
    )

    updated_plan, plan_progress = _compute_plan_progress(execution_plan, result.scratchpad)

    updates: dict[str, Any] = {
        "router_scratchpad": _scratchpad_to_dicts(result.scratchpad),
        "router_evidence": dict(result.evidence or {}),
        "router_iterations": result.iterations,
        "router_stop_reason": result.stop_reason,
        "execution_plan": updated_plan,
        "plan_progress": plan_progress,
        "router_validation_results": [
            dict(entry.validation or {})
            for entry in result.scratchpad
            if entry.validation
        ],
        "router_runtime_observations": [
            dict(entry.observation_data or {})
            for entry in result.scratchpad
            if entry.observation_data
        ],
        "runtime_audit_records": [
            audit
            for entry in result.scratchpad
            for audit in (entry.runtime_audit or [])
        ],
    }

    runtime_events = [
        build_runtime_event(
            result=entry.observation_data,
            turn=entry.turn,
            action=entry.action,
            plan_step_id=entry.plan_step_id,
            decision_source=entry.decision_source,
        )
        for entry in result.scratchpad
        if entry.observation_data
    ]
    executed_skill_calls = [
        runtime_result_to_skill_call(
            result=entry.observation_data,
            spec_manifest=_skill_manifest_for_event(registry, entry.skill_id),
            turn=entry.turn,
            arguments=dict(entry.arguments or {}),
        )
        for entry in result.scratchpad
        if entry.action == "call_skill" and entry.skill_id and entry.observation_data
    ]
    updates["runtime_events"] = runtime_events
    updates["runtime_summary"] = build_runtime_summary(runtime_events)
    updates["executed_skill_calls"] = executed_skill_calls

    # 把 skill 输出累计到 ops_evidence（沿用现有字段展示）
    skill_outputs = []
    for entry in result.scratchpad:
        if entry.action == "call_skill" and entry.status in {"done", "failed"}:
            skill_outputs.append(
                {
                    "skill_id": entry.skill_id,
                    "arguments": entry.arguments,
                    "result": entry.observation[:4000],
                    "status": entry.status,
                    "observation_data": dict(entry.observation_data or {}),
                    "decision_source": entry.decision_source,
                }
            )
    if skill_outputs:
        updates["ops_evidence"] = skill_outputs

    if result.pending_approval is not None:
        updates["approval_required"] = True
        updates["approval_payload"] = _build_approval_payload(result.pending_approval)
        updates["approval_status"] = "pending"
        updates["workflow_status"] = "waiting_approval"
        updates["resumable_from"] = "skill_router"
        updates["audit_trail"] = append_audit_event(
            state,
            "router_pending_approval",
            node="skill_router",
            details={
                "skill_id": result.pending_approval.skill_id,
                "risk": result.pending_approval.risk_level,
                "iterations": result.iterations,
            },
        )
        return updates

    # 正常结束
    final_answer = result.final_answer or "（router 未返回内容）"
    updates["final_answer"] = final_answer
    updates["approval_required"] = False
    updates["messages"] = [AIMessage(content=final_answer)]
    updates["workflow_status"] = "running"
    updates["audit_trail"] = append_audit_event(
        state,
        "router_finalized",
        node="skill_router",
        details={
            "stop_reason": result.stop_reason,
            "iterations": result.iterations,
            "skill_calls": len(skill_outputs),
        },
    )
    # 清空 approval_payload 以免影响后续
    if str((state.get("approval_payload") or {}).get("type") or "") == "skill_invocation":
        updates["approval_payload"] = {}
    return updates
