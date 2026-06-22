"""Unified Skill Router —— ReAct 风格的 LLM 驱动路由。

核心思路（对齐 openclaw）：
  1. 把注册表里所有 Skill 的元数据（description / when_to_use / argument_schema /
     example_invocations / risk_level）拼进 system prompt，让 LLM 看清"工具箱"。
  2. 主循环：LLM 观察 scratchpad + evidence → 决策（call_skill | finalize）→
     执行 skill（低危直接跑，medium/high 上报审批）→ 把 observation 塞回 scratchpad。
  3. 停止条件：LLM 主动说 finalize / 达到 max_iterations / 连续 3 次重复命中同一 skill+args。
  4. 审批：通过 `ApprovalRequest` 告诉上层图要 interrupt，恢复后把 approved/ rejected
     回写进 scratchpad 再继续循环（由 LangGraph 层负责 resume）。

这个模块**不依赖 LangGraph**，纯函数 + dataclass，方便单测。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage

from ops_rag_agent.prompts import apply_agent_persona
from ops_rag_agent.skills.registry import SkillRegistry
from ops_rag_agent.skills.runtime import (
    detect_duplicate_call,
    RuntimeAuditRecord,
    RuntimeErrorInfo,
    RuntimeResult,
    RuntimeStatus,
    ValidationResult,
    ValidationStatus,
    run_skill_runtime,
    skill_requires_approval,
)


# ---------- 常量 ----------

DEFAULT_MAX_ITERATIONS = 20
DUPLICATE_COMMAND_WINDOW = 3  # 连续 N 次命中同一 skill+args 就判死循环
_SHORTLIST_ALWAYS_INCLUDE = ("rag.search",)
_SHORTLIST_FALLBACK_IDS = ("rag.search", "web.search")
_SHORTLIST_HINT_DOMAIN_MAP = {
    "knowledge_base": ("knowledge", "rag"),
    "web_realtime": ("general", "web"),
    "ops_local": ("ops",),
    "ops_remote": ("ops",),
    "coding": ("platform", "general"),
    "general_qa": ("general",),
}


_SYSTEM_PROMPT_TEMPLATE = """你是一个运行在 Trae Agent 中的「统一 Skill Router」。
用户会给你一个自然语言请求，你有一组 Skill 工具可以调用。

你的工作流程（ReAct 风格）：
  1. 读取 USER_QUERY、INTENT_ANALYSIS（用户意图分析）、EXECUTION_PLAN（已规划好的步骤）、
     已有 EVIDENCE、SCRATCHPAD（历史 thought/action/observation）。
  2. **优先按 EXECUTION_PLAN 的步骤顺序执行**：每一步对应调用一个 skill。
     EXECUTION_PLAN 是建议性的——若实时观测到的证据表明计划不合理，可以动态调整或跳过某步。
  3. 思考下一步应当调用哪个 skill，或者是否已经可以得出结论。
  4. 严格输出 JSON，不要输出任何解释文字、不要 Markdown 代码块包裹。

可选的 JSON 输出格式：

# 情况 A：继续调用 skill
{{
  "thought": "中文说明你为什么选这个 skill，关联的 plan step（如果有）",
  "action": "call_skill",
  "skill_id": "<必须是 AVAILABLE_SKILLS 中的一个 id>",
  "arguments": {{...符合该 skill 的 argument_schema...}},
  "plan_step_id": "<可选：对应的 EXECUTION_PLAN 中某一步的 id>"
}}

# 情况 B：证据已充分，给出最终答案
{{
  "thought": "中文说明你为什么可以收束",
  "action": "finalize",
  "stop_reason": "enough_evidence" | "cannot_progress" | "user_request",
  "final_answer": "中文最终回复，可含分段结构（现状 / 判断 / 建议）"
}}

重要约束：
  - skill_id 必须严格等于 AVAILABLE_SKILLS 中的某个 id，不要自创
  - arguments 的 key/type 必须符合 argument_schema
  - 不要编造证据。没有观测到的事实，不要写入 final_answer
  - Runtime 会校验参数，并强制执行 rag.search 优先、审批拦截、循环熔断等规则
  - 若 observation 里出现 invalid_arguments / rag_search_required / approval_required /
    loop_detected 等错误码，你必须根据原因修正参数、先调用 rag.search，或换一个方案
  - 选 skill 的依据是每个 skill 的 when_to_use / description / risk_level，
    不要只看 skill_id 的字面值
  - 禁止仅凭用户"看起来像闲聊"就直接 finalize；若证据不足，应继续补充合适的 skill 观测

AVAILABLE_SKILLS（工具箱）:
{skill_catalog}

MAX_ITERATIONS: {max_iterations}
"""


# ---------- 数据结构 ----------


@dataclass
class ApprovalRequest:
    """需要上层图触发 interrupt 的审批请求。"""

    skill_id: str
    arguments: dict[str, Any]
    risk_level: str
    reason: str
    thought: str


@dataclass
class ScratchpadEntry:
    """一轮 ReAct 的 thought / action / observation。"""

    turn: int
    thought: str
    action: str  # "call_skill" | "finalize" | "approval_pending" | "error"
    skill_id: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    observation: str = ""  # skill 输出或错误信息
    status: str = "done"  # done / failed / pending_approval / skipped
    plan_step_id: str = ""  # 可选：对应 execution_plan.steps[i].id
    validation: dict[str, Any] = field(default_factory=dict)
    observation_data: dict[str, Any] = field(default_factory=dict)
    runtime_audit: list[dict[str, Any]] = field(default_factory=list)
    decision_source: str = ""


@dataclass
class RouterResult:
    final_answer: str
    stop_reason: str  # "finalize" | "max_iterations" | "loop_detected" | "pending_approval"
    iterations: int
    scratchpad: list[ScratchpadEntry]
    evidence: dict[str, Any] = field(default_factory=dict)
    pending_approval: Optional[ApprovalRequest] = None


@dataclass
class RouterAction:
    """统一的 Router 动作对象，兼容文本 JSON 与未来结构化 tool call 输入。"""

    action: str
    thought: str = ""
    skill_id: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    plan_step_id: str = ""
    stop_reason: str = ""
    final_answer: str = ""
    error: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)
    source: str = "json_text"

    def to_dict(self) -> dict[str, Any]:
        if self.action == "error":
            return {"action": "error", "error": self.error}
        data: dict[str, Any] = {"action": self.action}
        if self.thought:
            data["thought"] = self.thought
        if self.action == "call_skill":
            data["skill_id"] = self.skill_id
            data["arguments"] = dict(self.arguments)
            if self.plan_step_id:
                data["plan_step_id"] = self.plan_step_id
        elif self.action == "finalize":
            if self.stop_reason:
                data["stop_reason"] = self.stop_reason
            data["final_answer"] = self.final_answer
        return data


def _skipped_validation(
    arguments: dict[str, Any],
    *,
    model_name: str = "",
) -> ValidationResult:
    return ValidationResult(
        status=ValidationStatus.SKIPPED,
        model_name=model_name,
        raw_arguments=dict(arguments),
        normalized_arguments=dict(arguments),
    )


def _runtime_payload(result: RuntimeResult) -> tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    return (
        result.to_observation_text(),
        result.validation.to_dict(),
        result.to_dict(),
        [item.to_dict() for item in result.audit],
    )


def _scratchpad_to_runtime_history(scratchpad: list[ScratchpadEntry]) -> list[dict[str, Any]]:
    return [
        {
            "turn": entry.turn,
            "action": entry.action,
            "skill_id": entry.skill_id,
            "arguments": dict(entry.arguments or {}),
            "status": entry.status,
            "observation": entry.observation,
        }
        for entry in scratchpad
    ]


# ---------- 工具目录构造 ----------


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def shortlist_skill_specs(
    registry: SkillRegistry,
    *,
    allowed_business_domains: Optional[list[str]] = None,
    intent_analysis: Optional[dict[str, Any]] = None,
    execution_plan: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """基于意图/计划筛选当前轮次给 Router 的 skill shortlist。"""

    specs = registry.list_specs(allowed_business_domains=allowed_business_domains)
    if not specs:
        return []

    spec_by_id = {str(spec.get("skill_id") or ""): spec for spec in specs}
    selected_ids: list[str] = []
    selected_domains: list[str] = []
    hints = [
        str(item).strip()
        for item in (intent_analysis or {}).get("domain_hints", []) or []
        if str(item).strip()
    ]
    steps = (execution_plan or {}).get("steps") or []
    has_plan = any(isinstance(step, dict) for step in steps)

    for skill_id in _SHORTLIST_ALWAYS_INCLUDE:
        _append_unique(selected_ids, skill_id)

    for step in steps[:10]:
        if not isinstance(step, dict):
            continue
        skill_id = str(step.get("suggested_skill_id") or "").strip()
        _append_unique(selected_ids, skill_id)
        matched = spec_by_id.get(skill_id)
        if matched is not None:
            _append_unique(selected_domains, str(matched.get("business_domain") or ""))

    for hint in hints:
        for domain in _SHORTLIST_HINT_DOMAIN_MAP.get(hint, ()):
            _append_unique(selected_domains, domain)

    if not has_plan:
        for skill_id in _SHORTLIST_FALLBACK_IDS:
            _append_unique(selected_ids, skill_id)

    shortlisted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _include(spec: dict[str, Any]) -> None:
        skill_id = str(spec.get("skill_id") or "")
        if skill_id and skill_id not in seen_ids:
            shortlisted.append(spec)
            seen_ids.add(skill_id)

    for skill_id in selected_ids:
        matched = spec_by_id.get(skill_id)
        if matched is not None:
            _include(matched)

    if selected_domains:
        for spec in specs:
            if str(spec.get("business_domain") or "") in selected_domains:
                _include(spec)

    if not shortlisted:
        for skill_id in _SHORTLIST_FALLBACK_IDS:
            matched = spec_by_id.get(skill_id)
            if matched is not None:
                _include(matched)

    if not shortlisted:
        return list(specs)
    return shortlisted


def build_skill_catalog_block(
    registry: SkillRegistry,
    *,
    allowed_business_domains: Optional[list[str]] = None,
    selected_specs: Optional[list[dict[str, Any]]] = None,
) -> str:
    """把 SkillRegistry 里的元数据拼成给 LLM 看的文本目录。"""

    specs = (
        list(selected_specs)
        if selected_specs is not None
        else registry.list_specs(allowed_business_domains=allowed_business_domains)
    )
    lines: list[str] = []
    for spec in specs:
        sid = spec.get("skill_id", "")
        name = spec.get("name", sid)
        desc = spec.get("description", "")
        when = spec.get("when_to_use", "") or "(未填写)"
        risk = spec.get("risk_level", "low")
        req = "是" if spec.get("requires_approval") else "否"
        schema = spec.get("argument_schema") or {}
        examples = spec.get("example_invocations") or []
        lines.append(
            f"- skill_id: {sid}\n"
            f"  name: {name}\n"
            f"  description: {desc}\n"
            f"  when_to_use: {when}\n"
            f"  risk_level: {risk}  requires_approval: {req}\n"
            f"  argument_schema: {json.dumps(schema, ensure_ascii=False)}\n"
            f"  example_invocations: {json.dumps(examples, ensure_ascii=False)}"
        )
    return "\n".join(lines) if lines else "(no skills)"


def build_native_router_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "call_skill",
                "description": "Request the runtime to execute one registered skill.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "arguments": {"type": "object", "default": {}},
                        "plan_step_id": {"type": "string"},
                        "thought": {"type": "string"},
                    },
                    "required": ["skill_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finalize",
                "description": "Finalize the answer when enough evidence has been collected.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "final_answer": {"type": "string"},
                        "stop_reason": {"type": "string"},
                        "thought": {"type": "string"},
                    },
                    "required": ["final_answer"],
                },
            },
        },
    ]


# ---------- LLM 决策解析 ----------


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _normalize_action_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if "action" in payload:
        return dict(payload), "structured_action"

    name = str(payload.get("name") or payload.get("tool_name") or "").strip().lower()
    if name not in {"call_skill", "finalize"}:
        return {}, ""

    nested = payload.get("arguments")
    if not isinstance(nested, dict):
        nested = {}

    if name == "call_skill":
        skill_args = nested.get("arguments")
        if not isinstance(skill_args, dict):
            skill_args = (
                dict(nested)
                if nested and "skill_id" not in nested and "final_answer" not in nested
                else {}
            )
        return (
            {
                "action": "call_skill",
                "thought": str(payload.get("thought") or nested.get("thought") or ""),
                "skill_id": str(
                    payload.get("skill_id")
                    or nested.get("skill_id")
                    or payload.get("skill")
                    or ""
                ).strip(),
                "arguments": skill_args,
                "plan_step_id": str(
                    payload.get("plan_step_id") or nested.get("plan_step_id") or ""
                ).strip(),
            },
            "native_tool_call",
        )

    return (
        {
            "action": "finalize",
            "thought": str(payload.get("thought") or nested.get("thought") or ""),
            "stop_reason": str(
                payload.get("stop_reason") or nested.get("stop_reason") or ""
            ).strip(),
            "final_answer": str(
                payload.get("final_answer") or nested.get("final_answer") or ""
            ),
        },
        "native_tool_call",
    )


def parse_router_action(raw_response: Any) -> RouterAction:
    """把 Router 决策归一化成统一动作对象。"""

    if isinstance(raw_response, RouterAction):
        return raw_response

    tool_calls = getattr(raw_response, "tool_calls", None)
    if tool_calls:
        first_call = tool_calls[0]
        if isinstance(first_call, dict):
            payload = {
                "name": first_call.get("name"),
                "arguments": first_call.get("args") or first_call.get("arguments") or {},
            }
            if isinstance(raw_response, AIMessage) and isinstance(raw_response.content, str):
                payload["thought"] = raw_response.content
            action = parse_router_action(payload)
            action.source = "native_tool_call"
            return action

    if isinstance(raw_response, dict):
        data, source = _normalize_action_payload(raw_response)
        if not data:
            return RouterAction(
                action="error",
                error=f"unknown_action: {raw_response.get('action') or raw_response.get('name') or ''}",
                raw_payload=dict(raw_response),
                source="structured_action",
            )
    else:
        text = str(raw_response or "").strip()
        if not text:
            return RouterAction(action="error", error="empty_llm_output")
        # 兼容 ```json ... ``` 包裹
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
        match = _JSON_OBJECT_PATTERN.search(text)
        candidate = match.group(0) if match else text
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return RouterAction(action="error", error=f"json_decode_error: {exc}")
        if not isinstance(payload, dict):
            return RouterAction(action="error", error="decision_not_object")
        data, source = _normalize_action_payload(payload)
        if not data:
            action = str(payload.get("action") or payload.get("name") or "").strip().lower()
            return RouterAction(
                action="error",
                error=f"unknown_action: {action}",
                raw_payload=payload,
                source="json_text",
            )

    action = str(data.get("action") or "").strip().lower()
    if action not in {"call_skill", "finalize"}:
        return RouterAction(
            action="error",
            error=f"unknown_action: {action}",
            raw_payload=dict(data),
            source=source or "structured_action",
        )

    arguments = data.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}

    return RouterAction(
        action=action,
        thought=str(data.get("thought") or "").strip(),
        skill_id=str(data.get("skill_id") or "").strip(),
        arguments=dict(arguments),
        plan_step_id=str(data.get("plan_step_id") or "").strip(),
        stop_reason=str(data.get("stop_reason") or "").strip(),
        final_answer=str(data.get("final_answer") or ""),
        raw_payload=dict(data),
        source=source or "structured_action",
    )


def parse_router_decision(raw_text: Any) -> dict[str, Any]:
    """兼容旧接口：返回 dict 形态的 Router 决策。"""

    return parse_router_action(raw_text).to_dict()


# ---------- 审批门 ----------


def needs_approval(spec_manifest: dict[str, Any]) -> bool:
    """根据 skill manifest 决定是否强制审批。"""

    return skill_requires_approval(spec_manifest)


# ---------- 主循环 ----------


def run_skill_router(
    *,
    user_query: str,
    registry: SkillRegistry,
    llm_invoke: Callable[[str], Any],
    allowed_business_domains: Optional[list[str]] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    initial_evidence: Optional[dict[str, Any]] = None,
    initial_scratchpad: Optional[list[ScratchpadEntry]] = None,
    approval_resume: Optional[dict[str, Any]] = None,
    context_text: str = "",
    intent_analysis: Optional[dict[str, Any]] = None,
    execution_plan: Optional[dict[str, Any]] = None,
) -> RouterResult:
    """执行 ReAct 式的 skill 路由循环。

    参数：
      llm_invoke: 给一个 prompt，返回 LLM 文本输出。外部控制用哪个 LLM。
      approval_resume: 若非 None，说明当前是从一次 interrupt 恢复执行。
        形如 `{"approved": True|False, "skill_id": "...", "arguments": {...}}`
        approved=True 时，router 会直接跑这个 skill 作为本轮 observation；
        approved=False 时，把"被拒绝"写入 scratchpad 让 LLM 换策略。
      context_text: 历史对话摘要 + 最近原文（来自 memory/context），可空。
    """

    scratchpad: list[ScratchpadEntry] = list(initial_scratchpad or [])
    evidence: dict[str, Any] = dict(initial_evidence or {})

    # 先构造 skill_catalog block（整个循环共用）
    shortlisted_specs = shortlist_skill_specs(
        registry,
        allowed_business_domains=allowed_business_domains,
        intent_analysis=intent_analysis,
        execution_plan=execution_plan,
    )
    catalog_block = build_skill_catalog_block(
        registry,
        allowed_business_domains=allowed_business_domains,
        selected_specs=shortlisted_specs,
    )
    system_prompt = apply_agent_persona(
        _SYSTEM_PROMPT_TEMPLATE.format(
            skill_catalog=catalog_block,
            max_iterations=max_iterations,
        )
    )

    # 如果是审批恢复路径：先把上一步动作消化掉
    if approval_resume is not None:
        scratchpad.append(
            _consume_approval_resume(
                registry=registry,
                resume=approval_resume,
                evidence=evidence,
                history=_scratchpad_to_runtime_history(scratchpad),
                turn=len(scratchpad) + 1,
            )
        )

    for _ in range(max_iterations):
        turn = len(scratchpad) + 1
        prompt = _compose_prompt(
            system=system_prompt,
            user_query=user_query,
            evidence=evidence,
            scratchpad=scratchpad,
            context_text=context_text,
            intent_analysis=intent_analysis,
            execution_plan=execution_plan,
        )
        raw = llm_invoke(prompt)
        decision = parse_router_action(raw)
        thought = decision.thought

        if decision.action == "error":
            runtime_result = RuntimeResult(
                status=RuntimeStatus.FAILED,
                summary="router_decision_parse_failed",
                content=decision.error,
                error=RuntimeErrorInfo(
                    code="router_decision_parse_failed",
                    message=decision.error,
                ),
                audit=[
                    RuntimeAuditRecord(
                        phase="router_decision",
                        status="failed",
                        details={"raw_error": decision.error},
                    )
                ],
            )
            observation, validation, observation_data, runtime_audit = _runtime_payload(
                runtime_result
            )
            scratchpad.append(
                ScratchpadEntry(
                    turn=turn,
                    thought=thought or "(LLM 输出无法解析)",
                    action="error",
                    observation=observation,
                    status="failed",
                    validation=validation,
                    observation_data=observation_data,
                    runtime_audit=runtime_audit,
                    decision_source=decision.source,
                )
            )
            continue

        if decision.action == "finalize":
            scratchpad.append(
                ScratchpadEntry(
                    turn=turn,
                    thought=thought,
                    action="finalize",
                    observation=decision.final_answer,
                    status="done",
                    decision_source=decision.source,
                )
            )
            return RouterResult(
                final_answer=decision.final_answer,
                stop_reason=decision.stop_reason or "finalize",
                iterations=turn,
                scratchpad=scratchpad,
                evidence=evidence,
            )

        # action == "call_skill"
        skill_id = decision.skill_id
        arguments = dict(decision.arguments)
        plan_step_id = decision.plan_step_id

        runtime_result = _invoke_skill_safe(
            registry=registry,
            skill_id=skill_id,
            arguments=arguments,
            scratchpad=scratchpad,
            approval_granted=False,
        )
        observation, validation, observation_data, runtime_audit = _runtime_payload(
            runtime_result
        )
        if runtime_result.status == RuntimeStatus.PENDING_APPROVAL:
            pending = ApprovalRequest(
                skill_id=skill_id,
                arguments=dict(runtime_result.approval_request.get("arguments") or arguments),
                risk_level=str(runtime_result.approval_request.get("risk_level") or "medium"),
                reason=str(runtime_result.approval_request.get("reason") or "需要人工审批。"),
                thought=thought,
            )
            scratchpad.append(
                ScratchpadEntry(
                    turn=turn,
                    thought=thought,
                    action="approval_pending",
                    skill_id=skill_id,
                    arguments=dict(runtime_result.approval_request.get("arguments") or arguments),
                    observation=observation,
                    status="pending_approval",
                    plan_step_id=plan_step_id,
                    validation=validation,
                    observation_data=observation_data,
                    runtime_audit=runtime_audit,
                    decision_source=decision.source,
                )
            )
            return RouterResult(
                final_answer="",
                stop_reason="pending_approval",
                iterations=turn,
                scratchpad=scratchpad,
                evidence=evidence,
                pending_approval=pending,
            )

        status = "done"
        if runtime_result.status == RuntimeStatus.FAILED:
            status = "failed"
        elif runtime_result.status == RuntimeStatus.BLOCKED:
            status = "blocked"
        scratchpad.append(
            ScratchpadEntry(
                turn=turn,
                thought=thought,
                action="call_skill",
                skill_id=skill_id,
                arguments=arguments,
                observation=observation,
                status=status,
                plan_step_id=plan_step_id,
                validation=validation,
                observation_data=observation_data,
                runtime_audit=runtime_audit,
                decision_source=decision.source,
            )
        )
        evidence.setdefault("skill_outputs", []).append(
            {
                "skill_id": skill_id,
                "arguments": dict(arguments),
                "result": observation[:4000],
                "status": runtime_result.status.value,
                "structured_output": dict(runtime_result.structured_output),
                "error": runtime_result.error.to_dict() if runtime_result.error is not None else None,
            }
        )
        if (
            runtime_result.status == RuntimeStatus.BLOCKED
            and runtime_result.error is not None
            and runtime_result.error.code == "loop_detected"
        ):
            return RouterResult(
                final_answer="检测到 skill 路由死循环，已终止。请检查工具选择策略或扩充可用 skills。",
                stop_reason="loop_detected",
                iterations=turn,
                scratchpad=scratchpad,
                evidence=evidence,
            )

    # 达到 max_iterations，没 finalize → 给个兜底回复
    return RouterResult(
        final_answer="已达到最大迭代轮数但仍未收束，请基于上面累积的工具结果自行判断或拆分问题。",
        stop_reason="max_iterations",
        iterations=max_iterations,
        scratchpad=scratchpad,
        evidence=evidence,
    )


# ---------- 内部 helpers ----------


def _compose_prompt(
    *,
    system: str,
    user_query: str,
    evidence: dict[str, Any],
    scratchpad: list[ScratchpadEntry],
    context_text: str,
    intent_analysis: Optional[dict[str, Any]] = None,
    execution_plan: Optional[dict[str, Any]] = None,
) -> str:
    parts = [system]
    if context_text:
        parts.append("# 对话上下文（摘要 + 最近原文）\n" + context_text)
    parts.append(f"USER_QUERY:\n{user_query}")
    if intent_analysis:
        parts.append(
            "INTENT_ANALYSIS:\n"
            + json.dumps(intent_analysis, ensure_ascii=False, default=str)[:2000]
        )
    if execution_plan:
        # 给 plan 做一个简洁版摘要，防止 prompt 爆长
        plan_for_prompt = {
            "reasoning": execution_plan.get("reasoning", ""),
            "steps": [
                {
                    "id": s.get("id"),
                    "title": s.get("title"),
                    "intent": s.get("intent"),
                    "suggested_skill_id": s.get("suggested_skill_id"),
                    "suggested_arguments": s.get("suggested_arguments") or {},
                    "expected_output": s.get("expected_output"),
                    "status": s.get("status", "planned"),
                }
                for s in (execution_plan.get("steps") or [])[:10]
            ],
        }
        parts.append(
            "EXECUTION_PLAN:\n"
            + json.dumps(plan_for_prompt, ensure_ascii=False, default=str)[:4000]
        )
    parts.append(
        "EVIDENCE_JSON:\n"
        + json.dumps(evidence, ensure_ascii=False, default=str)[:6000]
    )
    parts.append(
        "SCRATCHPAD:\n"
        + json.dumps(
            [
                {
                    "turn": e.turn,
                    "thought": e.thought,
                    "action": e.action,
                    "skill_id": e.skill_id,
                    "arguments": e.arguments,
                    "observation": e.observation[:1500],
                    "status": e.status,
                    "plan_step_id": e.plan_step_id,
                }
                for e in scratchpad[-8:]  # 只给最近 8 轮，避免 prompt 爆长
            ],
            ensure_ascii=False,
        )
    )
    parts.append("请严格按要求输出 JSON 决策：")
    return "\n\n".join(parts)


def _invoke_skill_safe(
    *,
    registry: SkillRegistry,
    skill_id: str,
    arguments: dict[str, Any],
    scratchpad: list[ScratchpadEntry],
    approval_granted: bool,
) -> RuntimeResult:
    return run_skill_runtime(
        registry=registry,
        skill_id=skill_id,
        arguments=arguments,
        history=_scratchpad_to_runtime_history(scratchpad),
        approval_granted=approval_granted,
        enforce_rag_first=True,
        duplicate_window=DUPLICATE_COMMAND_WINDOW,
    )


def _is_looping(
    scratchpad: list[ScratchpadEntry],
    skill_id: str,
    arguments: dict[str, Any],
    *,
    window: int,
) -> bool:
    """最近 window 轮里，只要有同 skill_id + 同 args 的记录，就判定为死循环。"""

    return detect_duplicate_call(
        _scratchpad_to_runtime_history(scratchpad),
        skill_id=skill_id,
        arguments=arguments,
        window=window,
    )


def _consume_approval_resume(
    *,
    registry: SkillRegistry,
    resume: dict[str, Any],
    evidence: dict[str, Any],
    history: list[dict[str, Any]],
    turn: int,
) -> ScratchpadEntry:
    approved = bool(resume.get("approved"))
    skill_id = str(resume.get("skill_id") or "")
    arguments = resume.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}

    if not approved:
        runtime_result = RuntimeResult(
            status=RuntimeStatus.BLOCKED,
            skill_id=skill_id,
            summary="approval_rejected",
            content="user_rejected",
            error=RuntimeErrorInfo(
                code="approval_rejected",
                message="User rejected the skill invocation.",
            ),
            validation=_skipped_validation(arguments),
            audit=[
                RuntimeAuditRecord(
                    phase="approval",
                    status="blocked",
                    skill_id=skill_id,
                    details={"approved": False},
                )
            ],
        )
        observation, validation, observation_data, runtime_audit = _runtime_payload(
            runtime_result
        )
        return ScratchpadEntry(
            turn=turn,
            thought="用户拒绝了上一步 skill 调用",
            action="approval_rejected",
            skill_id=skill_id,
            arguments=arguments,
            observation=observation,
            status="skipped",
            validation=validation,
            observation_data=observation_data,
            runtime_audit=runtime_audit,
            decision_source="approval_resume",
        )

    runtime_result = run_skill_runtime(
        registry=registry,
        skill_id=skill_id,
        arguments=arguments,
        history=history,
        approval_granted=True,
        enforce_rag_first=True,
        duplicate_window=DUPLICATE_COMMAND_WINDOW,
    )
    observation, validation, observation_data, runtime_audit = _runtime_payload(
        runtime_result
    )
    status = "done" if runtime_result.status == RuntimeStatus.SUCCESS else "failed"
    evidence.setdefault("skill_outputs", []).append(
        {
            "skill_id": skill_id,
            "arguments": arguments,
            "result": observation[:4000],
            "status": runtime_result.status.value,
            "structured_output": dict(runtime_result.structured_output),
            "error": runtime_result.error.to_dict() if runtime_result.error is not None else None,
        }
    )
    return ScratchpadEntry(
        turn=turn,
        thought="用户已审批通过，执行上一步 skill",
        action="call_skill",
        skill_id=skill_id,
        arguments=arguments,
        observation=observation,
        status=status,
        validation=validation,
        observation_data=observation_data,
        runtime_audit=runtime_audit,
        decision_source="approval_resume",
    )
