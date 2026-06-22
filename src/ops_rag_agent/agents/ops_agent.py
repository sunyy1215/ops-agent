from __future__ import annotations

import json
import time
from langchain_core.messages import AIMessage

from ops_rag_agent.config import settings
from ops_rag_agent.guardrails import allow_guardrail_result, merge_guardrail_state, review_agent_guardrails
from ops_rag_agent.models.factory import build_chat_llm
from ops_rag_agent.observability import build_audit_event
from ops_rag_agent.ops.actions import actions_require_approval, build_remediation_actions
from ops_rag_agent.ops.diagnostics.local_health import diagnose_local_health
from ops_rag_agent.ops.diagnostics.local_health_llm import (
    llm_analyze_local_health,
    llm_recommend_local_health,
    llm_summarize_local_health,
)
from ops_rag_agent.ops.subagents import (
    DEFAULT_MAX_EXECUTION_SECONDS,
    DEFAULT_MAX_TOOL_CALLS,
    build_ops_worker_tasks,
    build_source_quality_policy,
    infer_execution_target_from_query,
    run_worker_task,
)
from ops_rag_agent.prompts import apply_agent_persona, get_prompt_spec
from ops_rag_agent.skills.registry import SkillRegistry


def run_ops_agent(state: dict, skill_registry: SkillRegistry) -> dict:
    query = state.get("user_query", "")
    if _looks_like_macos_local_health(query):
        return _run_macos_local_health_ops(state, skill_registry)

    prompt_spec = get_prompt_spec("ops.planner")
    anomaly_list = _detect_anomalies(query)
    execution_target, target_host = infer_execution_target_from_query(query)
    worker_tasks = build_ops_worker_tasks(
        anomaly_list,
        execution_target=execution_target,
        target_host=target_host,
    )
    worker_results = [run_worker_task(task, skill_registry) for task in worker_tasks]
    source_policy = build_source_quality_policy()
    catalog = skill_registry.grouped_specs(
        allowed_business_domains=settings.allowed_skill_business_domains
    )

    ops_evidence = _flatten_worker_records(worker_results, "evidence")
    ops_skill_calls = _flatten_worker_records(worker_results, "skill_calls")
    ops_failed_skills = _flatten_worker_records(worker_results, "failed_skills")
    planned_skill_calls = list(ops_skill_calls)
    executed_skill_calls = [
        call
        for call in ops_skill_calls
        if str(call.get("status", "")).strip() in {"done", "failed"}
    ]

    remediation_plan = _build_remediation_plan(
        anomaly_list=anomaly_list,
        execution_target=execution_target,
        target_host=target_host,
        evidence=ops_evidence,
    )

    remediation_actions = build_remediation_actions(
        query=query,
        execution_target=execution_target,
        target_host=target_host,
        remediation_plan=remediation_plan,
    )
    approval_required = actions_require_approval(remediation_actions) or any(
        "restart" in step.lower() or "rollback" in step.lower() or "sudo" in step.lower()
        for step in remediation_plan
    )
    budget_summary = _summarize_worker_budgets(worker_results)

    answer = _summarize_generic_ops_answer(
        query=query,
        execution_target=execution_target,
        target_host=target_host,
        anomaly_list=anomaly_list,
        worker_results=worker_results,
        ops_evidence=ops_evidence,
        ops_failed_skills=ops_failed_skills,
        remediation_plan=remediation_plan,
        remediation_actions=remediation_actions,
        approval_required=approval_required,
    )

    updates = {
        "anomaly_list": anomaly_list,
        "ops_worker_tasks": worker_results,
        "ops_execution_target": {
            "execution_target": execution_target,
            "target_host": target_host,
        },
        "ops_evidence": ops_evidence,
        "ops_skill_calls": ops_skill_calls,
        "ops_failed_skills": ops_failed_skills,
        # Expose skill calls for guardrails compatibility.
        "planned_skill_calls": planned_skill_calls,
        "executed_skill_calls": executed_skill_calls,
        "ops_budget_policy": {
            "max_tool_calls_per_worker": DEFAULT_MAX_TOOL_CALLS,
            "max_execution_seconds_per_worker": DEFAULT_MAX_EXECUTION_SECONDS,
            "worker_budget_summary": budget_summary,
        },
        "ops_source_policy": source_policy,
        "remediation_plan": remediation_plan,
        "remediation_actions": remediation_actions,
        "approval_required": approval_required,
        "approval_status": "pending" if approval_required else "not_required",
        "workflow_status": "waiting_approval" if approval_required else "running",
        "resumable_from": "approval_gate" if approval_required else "",
        "prompt_versions": {
            **state.get("prompt_versions", {}),
            "ops_agent": prompt_spec.version,
        },
        "approval_payload": {
            "reason": "remediation plan may affect system state",
            "registry_skills": catalog["all"],
            "skill_catalog": catalog,
            "ops_budget_policy": {
                "max_tool_calls_per_worker": DEFAULT_MAX_TOOL_CALLS,
                "max_execution_seconds_per_worker": DEFAULT_MAX_EXECUTION_SECONDS,
            },
            "ops_source_policy": source_policy,
            "actions": remediation_actions,
        },
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "available_skills": catalog["all"],
        "regular_skills": catalog["regular"],
        "complex_skills": catalog["complex_dev"],
        "skill_access_context": {
            "allowed_business_domains": list(settings.allowed_skill_business_domains),
            "ops_source_priority": list(source_policy["preferred_sources"]),
        },
    }
    guardrail_result = (
        review_agent_guardrails(
            route="ops",
            state=state,
            updates=updates,
            skill_registry=skill_registry,
            allowed_business_domains=settings.allowed_skill_business_domains,
            enable_model_review=settings.guardrails_model_review_enabled,
        )
        if settings.guardrails_enabled
        else allow_guardrail_result()
    )
    updates.update(
        merge_guardrail_state(
            state,
            guardrail_result,
            approval_payload=updates.get("approval_payload"),
            block_message="Ops remediation plan blocked by guardrails.",
        )
    )
    audit_trail = list(state.get("audit_trail", []) or [])
    for call in executed_skill_calls:
        args = call.get("arguments", {}) or {}
        audit_trail.append(
            build_audit_event(
                "skill_executed",
                node="ops_agent",
                details={
                    "skill_id": call.get("skill_id", ""),
                    "target_host": str(args.get("target_host", "") or ""),
                    "arguments": dict(args),
                    "duration_ms": int(call.get("duration_ms", 0) or 0),
                    "success": bool(call.get("status") == "done"),
                },
            )
        )
    audit_trail.append(
        build_audit_event(
            "ops_investigation_completed",
            node="ops_agent",
            details={
                "anomaly_count": len(anomaly_list),
                "worker_task_count": len(worker_results),
                "execution_target": execution_target,
                "target_host": target_host,
                "approval_required": updates.get("approval_required", False),
                "prompt_version": prompt_spec.version,
                "guardrail_action": updates.get("guardrail_action", "allow"),
                "budget_exhausted_worker_count": sum(
                    1 for item in worker_results if item.get("budget_exhausted")
                ),
                "ops_evidence_count": len(ops_evidence),
                "ops_skill_call_count": len(ops_skill_calls),
                "ops_failed_skill_count": len(ops_failed_skills),
                "action_count": len(remediation_actions),
                "remediation_action_count": sum(
                    1 for a in remediation_actions if a.get("category") == "remediation"
                ),
            },
        )
    )
    updates["audit_trail"] = audit_trail
    return updates


def _summarize_generic_ops_answer(
    *,
    query: str,
    execution_target: str,
    target_host: str,
    anomaly_list: list[str],
    worker_results: list[dict[str, Any]],
    ops_evidence: list[dict[str, Any]],
    ops_failed_skills: list[dict[str, Any]],
    remediation_plan: list[str],
    remediation_actions: list[dict[str, Any]],
    approval_required: bool,
) -> str:
    payload = {
        "query": query,
        "execution_target": execution_target,
        "target_host": target_host,
        "anomaly_list": anomaly_list,
        "worker_summaries": [
            {
                "worker_id": item.get("worker_id"),
                "title": item.get("title"),
                "findings": list(item.get("findings", []) or [])[:3],
                "recommendation": item.get("recommendation", ""),
                "budget_exhausted": bool(item.get("budget_exhausted", False)),
                "budget_exhausted_reason": item.get("budget_exhausted_reason", ""),
            }
            for item in worker_results
        ],
        "evidence": [
            {
                "source_type": item.get("source_type"),
                "summary": item.get("summary"),
                "status": item.get("status"),
                "result_excerpt": item.get("result_excerpt", ""),
            }
            for item in ops_evidence[:8]
        ],
        "failed_skills": ops_failed_skills,
        "remediation_plan": remediation_plan,
        "remediation_actions": remediation_actions,
        "approval_required": approval_required,
    }

    if _can_use_chat_llm():
        try:
            llm = build_chat_llm()
            prompt = (
                apply_agent_persona(
                    "你是一个中文运维助手。请基于给定的真实工具采集结果，输出一段最终回复，"
                    "不要解释内部实现过程，不要提 worker 预算，不要说自己没有采集数据。"
                    "要求："
                    "1. 明确说明当前观察到的系统状态；"
                    "2. 给出最可能的问题判断；"
                    "3. 给出 2-3 条可执行建议；"
                    "4. 如果需要审批，直接说明需要确认后才能继续；"
                    "5. 全程使用中文。"
                )
                + "\n\nINPUT_JSON:\n"
                + json.dumps(payload, ensure_ascii=False)
            )
            msg = llm.invoke(prompt)
            answer = getattr(msg, "content", str(msg)).strip()
            if answer:
                return answer
        except Exception:
            pass

    return _format_generic_ops_answer(
        execution_target=execution_target,
        target_host=target_host,
        ops_evidence=ops_evidence,
        remediation_plan=remediation_plan,
        approval_required=approval_required,
    )


def _format_generic_ops_answer(
    *,
    execution_target: str,
    target_host: str,
    ops_evidence: list[dict[str, Any]],
    remediation_plan: list[str],
    approval_required: bool,
) -> str:
    target_desc = "本机" if execution_target == "local" else f"远端主机 {target_host or '(未指定)'}"
    evidence_lines: list[str] = []
    for item in ops_evidence[:3]:
        summary = str(item.get("summary") or "").strip()
        result_excerpt = str(item.get("result_excerpt") or "").strip()
        if len(result_excerpt) > 120:
            result_excerpt = result_excerpt[:120] + "..."
        line = summary or str(item.get("source_type") or "未知来源")
        if result_excerpt:
            line += f"：{result_excerpt}"
        evidence_lines.append(f"- {line}")

    suggestions = [f"- {step}" for step in remediation_plan[:3] if str(step).strip()]
    if not suggestions:
        suggestions = ["- 建议先补充更具体的错误现象、时间范围和影响对象，再继续排查。"]

    lines = [
        f"已基于 {target_desc} 的真实采集结果完成本轮排查。",
        "当前观察：",
        *(evidence_lines or ["- 暂未拿到足够的高质量证据。"]),
        "建议：",
        *suggestions,
    ]
    if approval_required:
        lines.append("说明：后续建议中包含可能影响系统状态的动作，需要你确认后才能继续执行。")
    return "\n".join(lines)


def _can_use_chat_llm() -> bool:
    api_key = str(settings.llm_api_key or "").strip()
    return bool(api_key and api_key != "change-me")


def _looks_like_macos_local_health(query: str) -> bool:
    q = (query or "").lower()
    markers = (
        "powermetrics",
        "thermal",
        "temperature",
        "thermals",
        "fan",
        "rpm",
        "功耗",
        "频率",
        "风扇",
        "温度",
        "热",
        "热压力",
        "降频",
        "gpu",
        "显卡",
        # Terminal / shell intents always go through the macOS local loop
        # so that ops.bash.readonly is available.
        "终端",
        "命令",
        "命令行",
        "shell",
        "bash",
        "执行命令",
        "跑一下",
        "跑个",
        "排查",
        "诊断",
        "进程",
        "端口",
    )
    local_context_markers = ("mac", "macos", "本机", "电脑", "机器", "当前机器", "这台")
    local_health_markers = ("cpu", "memory", "内存", "swap", "卡", "卡顿", "日志", "log", "磁盘", "disk")
    return any(m in q for m in markers) or (
        any(m in q for m in local_context_markers)
        and any(m in q for m in local_health_markers)
    )


def _run_macos_local_health_ops(state: dict, skill_registry: SkillRegistry) -> dict:
    """
    Local-only closed loop:
    observe -> analyze -> verify -> recommend

    - Collects evidence via macOS-only skills.
    - Allows sudo powermetrics, but requires explicit collection authorization.
    - Never executes remediation actions; only outputs suggestions and user-runnable commands.
    """

    prompt_spec = get_prompt_spec("ops.planner")
    query = str(state.get("user_query", "") or "")
    catalog = skill_registry.grouped_specs(
        allowed_business_domains=settings.allowed_skill_business_domains
    )

    ops_skill_calls: list[dict] = list(state.get("ops_skill_calls", []) or [])
    ops_failed_skills: list[dict] = list(state.get("ops_failed_skills", []) or [])
    ops_evidence: list[dict] = list(state.get("ops_evidence", []) or [])
    ops_execution_steps: list[dict] = list(state.get("ops_execution_steps", []) or [])
    iteration_count = int(state.get("ops_iteration_count", 0) or 0)
    iteration_limit = int(state.get("ops_iteration_limit", 10) or 10)
    approval_payload = state.get("approval_payload", {}) or {}
    approved_iteration_commands = (
        list(approval_payload.get("commands", []) or [])
        if state.get("approval_status") == "approved"
        and str(approval_payload.get("type", "")).strip() == "readonly_ops_iteration"
        else []
    )
    fast_iteration_mode = bool(approved_iteration_commands)
    current_turn = max(
        [
            int(item.get("turn") or 0)
            for item in [*ops_skill_calls, *ops_execution_steps]
            if isinstance(item, dict)
        ],
        default=0,
    )

    def next_turn() -> int:
        nonlocal current_turn
        current_turn += 1
        return current_turn

    def call_skill(
        skill_id: str,
        arguments: dict,
        *,
        turn: int,
        title: str | None = None,
        summary: str | None = None,
    ) -> str:
        start = time.monotonic()
        result_text = ""
        status = "done"
        success = True
        try:
            skill = skill_registry.get(skill_id)
            result_text = str(skill.invoke(arguments))
            if result_text.startswith(("SECURITY_ERROR", "APPROVAL_REQUIRED", "ERROR")):
                status = "failed"
                success = False
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            success = False
            result_text = f"ERROR: {type(exc).__name__}: {exc}"

        duration_ms = int((time.monotonic() - start) * 1000)
        command = str(arguments.get("cmd", "")).strip() or None
        result_excerpt = _extract_result_excerpt(result_text)
        ops_skill_calls.append(
            {
                "turn": turn,
                "skill_id": skill_id,
                "arguments": dict(arguments),
                "version": "",
                "business_domain": "ops",
                "kind": "regular",
                "requires_approval": False,
                "status": status,
                "result": result_text[:8000],
                "result_excerpt": result_excerpt,
                "duration_ms": duration_ms,
                "success": success,
            }
        )
        ops_execution_steps.append(
            {
                "turn": turn,
                "title": title or skill_id,
                "summary": summary or _default_step_summary(skill_id, command, success),
                "skill_id": skill_id,
                "status": status,
                "success": success,
                "duration_ms": duration_ms,
                "command": command,
                "result_excerpt": result_excerpt,
            }
        )
        if status == "failed":
            ops_failed_skills.append(
                {
                    "skill_id": skill_id,
                    "reason": result_text[:240],
                }
            )
        return result_text

    evidence_struct = {
        "metrics": _find_latest_evidence_payload(ops_evidence, "macos.metrics"),
        "gpu_profile": _find_latest_evidence_payload(ops_evidence, "macos.gpu_profile"),
        "powermetrics": _find_latest_evidence_payload(ops_evidence, "macos.powermetrics"),
    }

    # ---- observe
    if not fast_iteration_mode:
        observe_turn = next_turn()
        metrics_text = call_skill(
            "ops.macos.metrics",
            {"top_n": 10},
            turn=observe_turn,
            title="采集本机指标",
            summary="读取 CPU、内存、Swap 和高占用进程快照。",
        )
        gpu_profile_text = call_skill(
            "ops.macos.gpu_profile",
            {},
            turn=observe_turn,
            title="采集 GPU 信息",
            summary="读取本机 GPU 和显示上下文信息。",
        )

        authorized = bool(state.get("macos_powermetrics_authorized", False))
        if not authorized and state.get("require_powermetrics_approval", False):
            # Ask for explicit collection authorization to run sudo powermetrics.
            planned_skill_calls = [
                {
                    "turn": observe_turn,
                    "skill_id": "ops.macos.powermetrics",
                    "arguments": {
                        "approved": True,
                        "show_process_energy": False,
                    },
                    "business_domain": "ops",
                    "requires_approval": True,
                }
            ]
            approval_payload = {
                "type": "collect_powermetrics",
                "reason": "Collect read-only thermals/power/frequency/fan telemetry via sudo powermetrics.",
                "requested_command_family": [
                    "sudo -n /usr/bin/powermetrics --samplers smc -n 1",
                    "sudo -n /usr/bin/powermetrics --samplers smc -n 1 --show-process-energy",
                ],
                "safety_notes": [
                    "No shell=True; fixed argv only.",
                    "sudo uses -n (non-interactive); will fail fast if not authorized/cached.",
                    "Output is collected as raw + parsed fields for auditability.",
                ],
                "registry_skills": catalog["all"],
                "skill_catalog": catalog,
                "planned_skill_calls": planned_skill_calls,
            }
            answer = (
                "需要一次采集授权：为了采集温度/热压力/功耗/频率/风扇信息，需要运行只读的 "
                "`sudo -n /usr/bin/powermetrics --samplers smc -n 1`。批准后会继续完成诊断并只输出建议。"
            )
            return {
                "route": "ops",
                "ops_execution_target": {"execution_target": "local", "target_host": ""},
                "ops_worker_tasks": [],
                "ops_evidence": ops_evidence,
                "ops_skill_calls": ops_skill_calls,
                "ops_execution_steps": ops_execution_steps,
                "ops_failed_skills": ops_failed_skills,
                "planned_skill_calls": planned_skill_calls,
                "executed_skill_calls": [],
                "remediation_actions": [],
                "approval_required": True,
                "approval_status": "pending",
                "workflow_status": "waiting_approval",
                "resumable_from": "approval_gate",
                "approval_payload": approval_payload,
                "prompt_versions": {**state.get("prompt_versions", {}), "ops_agent": prompt_spec.version},
                "messages": [AIMessage(content=answer)],
                "final_answer": answer,
                "available_skills": catalog["all"],
                "regular_skills": catalog["regular"],
                "complex_skills": catalog["complex_dev"],
                "skill_access_context": {
                    "allowed_business_domains": list(settings.allowed_skill_business_domains),
                },
            }

        powermetrics_text = call_skill(
            "ops.macos.powermetrics",
            {
                "approved": True,
                "show_process_energy": False,
            },
            turn=observe_turn,
            title="采集温度与热状态",
            summary="读取温度、功耗、热压力和风扇等只读数据。",
        )

        evidence_struct = {
            "metrics": _safe_json(metrics_text),
            "gpu_profile": _safe_json(gpu_profile_text),
            "powermetrics": _safe_json(powermetrics_text),
        }
        ops_evidence.extend(
            [
                {"source_type": "macos.metrics", "summary": "Local metrics snapshot", "payload": evidence_struct["metrics"]},
                {"source_type": "macos.gpu_profile", "summary": "GPU/display context", "payload": evidence_struct["gpu_profile"]},
                {"source_type": "macos.powermetrics", "summary": "Thermals/power/fans", "payload": evidence_struct["powermetrics"]},
            ]
        )

    if fast_iteration_mode and not any(evidence_struct.values()):
        observe_turn = next_turn()
        metrics_text = call_skill(
            "ops.macos.metrics",
            {"top_n": 10},
            turn=observe_turn,
            title="补采本机指标",
            summary="上一轮基础证据缺失，本轮补采 CPU、内存和进程快照。",
        )
        evidence_struct["metrics"] = _safe_json(metrics_text)
        ops_evidence.append(
            {"source_type": "macos.metrics", "summary": "Local metrics snapshot", "payload": evidence_struct["metrics"]}
        )

    # powermetrics 采集授权门已合并到"只读命令迭代审批"中，这里默认视为已授权，
    # 避免用户连续经历两次审批。如需强制单独授权 sudo powermetrics，可通过
    # state["require_powermetrics_approval"] = True 重新开启。
    if (
        state.get("require_powermetrics_approval", False)
        and not state.get("macos_powermetrics_authorized", False)
        and not fast_iteration_mode
    ):
        # Ask for explicit collection authorization to run sudo powermetrics.
        planned_skill_calls = [
            {
                "turn": next_turn(),
                "skill_id": "ops.macos.powermetrics",
                "arguments": {
                    "approved": True,
                    "show_process_energy": False,
                },
                "business_domain": "ops",
                "requires_approval": True,
            }
        ]
        approval_payload = {
            "type": "collect_powermetrics",
            "reason": "Collect read-only thermals/power/frequency/fan telemetry via sudo powermetrics.",
            "requested_command_family": [
                "sudo -n /usr/bin/powermetrics --samplers smc -n 1",
                "sudo -n /usr/bin/powermetrics --samplers smc -n 1 --show-process-energy",
            ],
            "safety_notes": [
                "No shell=True; fixed argv only.",
                "sudo uses -n (non-interactive); will fail fast if not authorized/cached.",
                "Output is collected as raw + parsed fields for auditability.",
            ],
            "registry_skills": catalog["all"],
            "skill_catalog": catalog,
            "planned_skill_calls": planned_skill_calls,
        }
        answer = (
            "需要一次采集授权：为了采集温度/热压力/功耗/频率/风扇信息，需要运行只读的 "
            "`sudo -n /usr/bin/powermetrics --samplers smc -n 1`。批准后会继续完成诊断并只输出建议。"
        )
    if approved_iteration_commands:
        approved_turn = next_turn()
        # 把上一轮遗留的 planned 占位步骤清掉（同 cmd），避免前端出现
        # "planned + done" 两条同名记录。
        approved_cmds = {str(item.get("cmd") or "") for item in approved_iteration_commands}
        ops_execution_steps = [
            step
            for step in ops_execution_steps
            if not (
                str(step.get("status") or "") == "planned"
                and str(step.get("command") or "") in approved_cmds
            )
        ]
        for item in approved_iteration_commands:
            shell_text = call_skill(
                "ops.bash.readonly",
                {"cmd": item["cmd"]},
                turn=approved_turn,
                title=str(item.get("title", "执行只读命令")),
                summary=str(item.get("summary", "执行用户已批准的只读命令。")),
            )
            shell_payload = _safe_json(shell_text)
            ops_evidence.append(
                {
                    "source_type": "ops.bash.readonly",
                    "summary": str(item.get("summary", "执行用户已批准的只读命令。")),
                    "payload": shell_payload,
                }
            )
            evidence_struct.setdefault("shell_followups", []).append(shell_payload)
        iteration_count += 1

    # ---- analyze
    llm_audit: dict[str, Any] = {}
    if fast_iteration_mode:
        diagnosis = diagnose_local_health(evidence=evidence_struct)
        hypotheses = list(diagnosis.get("hypotheses", []) or [])
        llm_audit["analyze"] = {"ok": False, "skipped": True, "reason": "fast_iteration_mode"}
    else:
        llm_analyze = llm_analyze_local_health(evidence=evidence_struct)
        if llm_analyze.ok:
            hypotheses = list(llm_analyze.value.get("hypotheses", []) or [])
            llm_audit["analyze"] = {"ok": True}
        else:
            diagnosis = diagnose_local_health(evidence=evidence_struct)
            hypotheses = list(diagnosis.get("hypotheses", []) or [])
            llm_audit["analyze"] = {"ok": False, "error": llm_analyze.error}

    # ---- verify
    # Verify trigger uses either rule-engine summary or a simple heuristic when LLM is used.
    verify_suggested = False
    if not fast_iteration_mode and llm_analyze.ok:
        # If any hypothesis asks for timeseries, run it.
        verify_suggested = any(
            "timeseries" in " ".join(h.get("next_checks", [])).lower() for h in hypotheses
        )
    else:
        verify_suggested = bool(diagnosis.get("summary", {}).get("verify_suggested"))

    if verify_suggested:
        verify_turn = next_turn()
        ts_text = call_skill(
            "ops.macos.timeseries_probe",
            {
                "duration_s": 60,
                "metrics_interval_s": 5,
                "powermetrics_interval_s": 15,
                "include_powermetrics": True,
                "powermetrics_approved": True,
            },
            turn=verify_turn,
            title="执行短时序探针",
            summary="进行 60 秒短时序采样，确认异常是否持续存在。",
        )
        evidence_struct["timeseries"] = _safe_json(ts_text)
        ops_evidence.append(
            {
                "source_type": "macos.timeseries_probe",
                "summary": "Short timeseries probe",
                "payload": evidence_struct["timeseries"],
            }
        )
        # Re-run analyze once with additional evidence.
        llm_analyze2 = llm_analyze_local_health(evidence=evidence_struct)
        if llm_analyze2.ok:
            hypotheses = list(llm_analyze2.value.get("hypotheses", []) or [])
            llm_audit["analyze_after_verify"] = {"ok": True}
        else:
            diagnosis = diagnose_local_health(evidence=evidence_struct)
            hypotheses = list(diagnosis.get("hypotheses", []) or [])
            llm_audit["analyze_after_verify"] = {
                "ok": False,
                "error": llm_analyze2.error,
            }

    executed_commands = {
        str(call.get("arguments", {}).get("cmd", "")).strip()
        for call in ops_skill_calls
        if str(call.get("skill_id", "")).strip() == "ops.bash.readonly"
    }
    followup_commands = _build_local_followup_commands(
        query=query,
        evidence=evidence_struct,
        hypotheses=hypotheses,
        executed_commands=executed_commands,
    )
    if followup_commands and iteration_count < iteration_limit:
        followup_turn = next_turn()
        # 按 risk 分流：low 直接自动执行；medium/high 走审批弹窗
        low_risk_commands = [
            item for item in followup_commands
            if str(item.get("risk") or "").lower() == "low"
        ]
        elevated_commands = [
            item for item in followup_commands
            if str(item.get("risk") or "").lower() != "low"
        ]

        # 低危命令：立即执行，不触发 interrupt
        if low_risk_commands:
            for item in low_risk_commands:
                shell_text = call_skill(
                    "ops.bash.readonly",
                    {"cmd": item["cmd"]},
                    turn=followup_turn,
                    title=str(item.get("title", "执行只读命令")),
                    summary=str(item.get("summary", "自动执行低风险只读命令。")),
                )
                shell_payload = _safe_json(shell_text)
                ops_evidence.append(
                    {
                        "source_type": "ops.bash.readonly",
                        "summary": str(item.get("summary", "自动执行低风险只读命令。")),
                        "payload": shell_payload,
                    }
                )
                evidence_struct.setdefault("shell_followups", []).append(shell_payload)
            iteration_count += 1

        # 只有中/高危命令才需要审批
        if elevated_commands:
            for item in elevated_commands:
                ops_execution_steps.append(
                    {
                        "turn": followup_turn,
                        "title": item["title"],
                        "summary": item["summary"],
                        "skill_id": "ops.bash.readonly",
                        "status": "planned",
                        "success": True,
                        "duration_ms": 0,
                        "command": item["cmd"],
                        "result_excerpt": "待审批后执行。",
                        "risk": str(item.get("risk") or "medium"),
                    }
                )

            answer = _format_iteration_approval_answer(
                iteration_count=iteration_count,
                iteration_limit=iteration_limit,
                commands=elevated_commands,
                hypotheses=hypotheses,
                auto_executed=low_risk_commands,
            )
            return {
                "route": "ops",
                "ops_execution_target": {"execution_target": "local", "target_host": ""},
                "ops_worker_tasks": [],
                "ops_evidence": ops_evidence,
                "ops_skill_calls": ops_skill_calls,
                "ops_execution_steps": ops_execution_steps,
                "ops_failed_skills": ops_failed_skills,
                "planned_skill_calls": [
                    {
                        "turn": followup_turn,
                        "skill_id": "ops.bash.readonly",
                        "arguments": {"cmd": item["cmd"]},
                        "business_domain": "ops",
                        "requires_approval": True,
                        "risk": str(item.get("risk") or "medium"),
                        "status": "planned",
                    }
                    for item in elevated_commands
                ],
                "executed_skill_calls": [c for c in ops_skill_calls if c.get("status") in {"done", "failed"}],
                "ops_hypotheses": hypotheses,
                "ops_recommendations": list(state.get("ops_recommendations", []) or []),
                "remediation_actions": [],
                "approval_required": True,
                "approval_status": "pending",
                "workflow_status": "waiting_approval",
                "resumable_from": "approval_gate",
                "approval_payload": {
                    "type": "readonly_ops_iteration",
                    "reason": f"第 {iteration_count + 1} 轮存在中/高风险命令，请审批",
                    "iteration_count": iteration_count,
                    "iteration_limit": iteration_limit,
                    "commands": elevated_commands,
                    "auto_executed_low_risk": [
                        {
                            "title": item["title"],
                            "cmd": item["cmd"],
                            "risk": "low",
                            "turn": followup_turn,
                        }
                        for item in low_risk_commands
                    ],
                    "actions": [
                        {
                            "action_id": f"readonly-iter-{iteration_count + 1}-{idx + 1}",
                            "category": "readonly_diagnosis",
                            "title": item["title"],
                            "description": item["summary"],
                            "command": item["cmd"],
                            "risk": str(item.get("risk") or "medium"),
                            "turn": followup_turn,
                        }
                        for idx, item in enumerate(elevated_commands)
                    ],
                },
                "prompt_versions": {**state.get("prompt_versions", {}), "ops_agent": prompt_spec.version},
                "messages": [AIMessage(content=answer)],
                "final_answer": answer,
                "available_skills": catalog["all"],
                "regular_skills": catalog["regular"],
                "complex_skills": catalog["complex_dev"],
                "skill_access_context": {
                    "allowed_business_domains": list(settings.allowed_skill_business_domains),
                },
                "ops_iteration_count": iteration_count,
                "ops_iteration_limit": iteration_limit,
            }
        # 若本轮全是低危：跳过 interrupt，继续走下面的"总结/建议"段落


    # ---- recommend (LLM preferred)
    llm_reco = llm_recommend_local_health(evidence=evidence_struct, hypotheses=hypotheses)
    if llm_reco.ok:
        recommendations = list(llm_reco.value.get("recommendations", []) or [])
        llm_audit["recommend"] = {"ok": True}
        # Keep a minimal summary from rule engine for consistent formatting.
        summary = diagnose_local_health(evidence=evidence_struct).get("summary", {}) or {}
    else:
        diagnosis = diagnose_local_health(evidence=evidence_struct)
        recommendations = list(diagnosis.get("recommendations", []) or [])
        summary = diagnosis.get("summary", {}) or {}
        llm_audit["recommend"] = {"ok": False, "error": llm_reco.error}

    # ---- final answer synthesis (LLM preferred, template fallback)
    # 回灌历史摘要 + 最近原文，让最终回答具备跨轮上下文连贯性
    from ops_rag_agent.memory.context import build_conversation_context

    context_text = build_conversation_context(
        state,
        tail_messages=settings.history_tail_messages,
        exclude_last_user=True,
    )
    llm_summary = llm_summarize_local_health(
        summary=summary,
        hypotheses=hypotheses,
        recommendations=recommendations,
        context_text=context_text or None,
    )
    if llm_summary.ok:
        answer = str(llm_summary.value.get("answer", "")).strip()
        llm_audit["final_answer"] = {"ok": True}
    else:
        answer = _format_local_health_answer(
            summary=summary,
            hypotheses=hypotheses,
            recommendations=recommendations,
        )
        llm_audit["final_answer"] = {"ok": False, "error": llm_summary.error}

    updates = {
        "route": "ops",
        "ops_execution_target": {"execution_target": "local", "target_host": ""},
        "ops_worker_tasks": [],
        "ops_evidence": ops_evidence,
        "ops_skill_calls": ops_skill_calls,
        "ops_execution_steps": ops_execution_steps,
        "ops_failed_skills": ops_failed_skills,
        "planned_skill_calls": list(ops_skill_calls),
        "executed_skill_calls": [c for c in ops_skill_calls if c.get("status") in {"done", "failed"}],
        "ops_hypotheses": hypotheses,
        "ops_recommendations": recommendations,
        "remediation_actions": [],
        "approval_required": False,
        "approval_status": state.get("approval_status", "not_required"),
        "workflow_status": "running",
        "resumable_from": "",
        "approval_payload": {},
        "prompt_versions": {**state.get("prompt_versions", {}), "ops_agent": prompt_spec.version},
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "available_skills": catalog["all"],
        "regular_skills": catalog["regular"],
        "complex_skills": catalog["complex_dev"],
        "skill_access_context": {
            "allowed_business_domains": list(settings.allowed_skill_business_domains),
        },
        "ops_iteration_count": iteration_count,
        "ops_iteration_limit": iteration_limit,
    }

    guardrail_result = (
        review_agent_guardrails(
            route="ops",
            state=state,
            updates=updates,
            skill_registry=skill_registry,
            allowed_business_domains=settings.allowed_skill_business_domains,
            enable_model_review=settings.guardrails_model_review_enabled,
        )
        if settings.guardrails_enabled
        else allow_guardrail_result()
    )
    updates.update(
        merge_guardrail_state(
            state,
            guardrail_result,
            approval_payload=updates.get("approval_payload"),
            block_message="Ops request blocked by guardrails.",
        )
    )

    audit_trail = list(state.get("audit_trail", []) or [])
    audit_trail.append(
        build_audit_event(
            "ops_local_health_completed",
            node="ops_agent",
            details={
                "skill_call_count": len(ops_skill_calls),
                "execution_step_count": len(ops_execution_steps),
                "failed_skill_count": len(ops_failed_skills),
                "hypothesis_count": len(hypotheses),
                "recommendation_count": len(recommendations),
                "powermetrics_authorized": True,
                "llm": llm_audit,
            },
        )
    )
    updates["audit_trail"] = audit_trail
    return updates


def _safe_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        return {"raw": (text or "")[:8000]}


def _find_latest_evidence_payload(evidence_items: list[dict], source_type: str) -> dict:
    for item in reversed(list(evidence_items or [])):
        if str(item.get("source_type", "")).strip() != source_type:
            continue
        payload = item.get("payload")
        if isinstance(payload, dict):
            return payload
    return {}


def _extract_result_excerpt(text: str) -> str:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            stdout = str(parsed.get("stdout", "") or "").strip()
            stderr = str(parsed.get("stderr", "") or "").strip()
            if stdout:
                return stdout[:280]
            if stderr:
                return stderr[:280]
    except Exception:
        pass
    return (text or "").strip()[:280]


def _default_step_summary(skill_id: str, command: str | None, success: bool) -> str:
    if skill_id == "ops.bash.readonly" and command:
        prefix = "已执行只读命令" if success else "只读命令执行失败"
        return f"{prefix}: {command}"
    if skill_id == "ops.macos.metrics":
        return "采集本机 CPU、内存和进程快照。"
    if skill_id == "ops.macos.gpu_profile":
        return "采集本机 GPU 和显示信息。"
    if skill_id == "ops.macos.powermetrics":
        return "采集本机温度、热压力和功耗数据。"
    if skill_id == "ops.macos.timeseries_probe":
        return "执行短时序探针，确认异常趋势。"
    return f"执行 {skill_id}"


def _build_local_followup_commands(
    *,
    query: str,
    evidence: dict,
    hypotheses: list[dict],
    executed_commands: set[str],
) -> list[dict[str, str]]:
    """生成下一轮只读排查命令。

    策略：把常用 macOS 只读排查命令按"高/中/低"优先级排开，先放"与当前证据/查询强相关"
    的命令，再追加通用深入排查命令。每轮从未执行的命令里取前 2 条，直到池耗尽或达到
    迭代上限。每条命令都会带上 `risk` 字段（low/medium/high）：
      - low：纯只读、输出小且稳定（ps/vm_stat/sysctl/uptime/df/sw_vers/lsof LISTEN）
      - medium：采样/较大输出/连接统计（iostat/netstat、一般 error 日志）
      - high：涉及 panic、崩溃等敏感线索或需特权的命令
    调用方据此决定"低危直接执行、中高危走审批"。
    """

    metrics = evidence.get("metrics", {}) if isinstance(evidence.get("metrics"), dict) else {}
    cpu = metrics.get("cpu", {}) if isinstance(metrics.get("cpu"), dict) else {}
    memory = metrics.get("memory", {}) if isinstance(metrics.get("memory"), dict) else {}
    swap = metrics.get("swap", {}) if isinstance(metrics.get("swap"), dict) else {}

    cpu_percent = float(cpu.get("percent_total") or 0.0)
    mem_percent = float(memory.get("percent") or 0.0)
    swap_percent = float(swap.get("percent") or 0.0)

    lower_query = (query or "").lower()
    next_checks = " ".join(
        " ".join(item.get("next_checks", []) or []) for item in hypotheses if isinstance(item, dict)
    ).lower()

    mem_focus = (
        mem_percent >= 75 or swap_percent >= 10 or "memory" in next_checks or "内存" in lower_query
    )
    cpu_focus = (
        cpu_percent >= 70
        or "cpu" in next_checks
        or "卡" in lower_query
        or "降频" in lower_query
    )
    log_focus = "日志" in lower_query or "log" in lower_query
    net_focus = "网络" in lower_query or "端口" in lower_query or "network" in next_checks
    disk_focus = "磁盘" in lower_query or "disk" in lower_query or "io" in next_checks

    pool: list[dict[str, str]] = []

    # ---- 高优先级：与当前证据/查询强相关 ----
    if mem_focus:
        pool += [
            {
                "title": "排查高内存进程",
                "summary": "按内存占用倒序读取前 20 个进程。",
                "cmd": "ps -Ao pid,comm,%cpu,%mem,rss -m | head -20",
                "risk": "low",
            },
            {
                "title": "检查虚拟内存状态",
                "summary": "读取 vm_stat，判断是否存在持续 swap 压力。",
                "cmd": "vm_stat",
                "risk": "low",
            },
            {
                "title": "查看 swap 使用情况",
                "summary": "读取 swap 使用总量和压力。",
                "cmd": "sysctl vm.swapusage",
                "risk": "low",
            },
        ]
    if cpu_focus:
        pool += [
            {
                "title": "排查高 CPU 进程",
                "summary": "按 CPU 占用倒序读取前 20 个进程。",
                "cmd": "ps -Ao pid,comm,%cpu,%mem,time -r | head -20",
                "risk": "low",
            },
            {
                "title": "检查系统负载",
                "summary": "读取系统负载和运行时间。",
                "cmd": "uptime",
                "risk": "low",
            },
            {
                "title": "查看 CPU 核心数和型号",
                "summary": "确认物理/逻辑核心数，辅助判断并发压力。",
                "cmd": "sysctl -n hw.ncpu hw.logicalcpu hw.physicalcpu",
                "risk": "low",
            },
        ]
    if log_focus:
        pool += [
            {
                "title": "读取最近系统 error 日志",
                "summary": "读取最近 1 分钟 error 级别的系统日志。",
                "cmd": 'log show --last 1m --predicate eventMessage CONTAINS[c] "error" --style compact',
                "risk": "medium",
            },
            {
                "title": "读取最近 panic/崩溃线索",
                "summary": "读取最近 5 分钟关键字包含 panic/crash 的日志。",
                "cmd": 'log show --last 5m --predicate eventMessage CONTAINS[c] "panic" --style compact',
                "risk": "high",
            },
        ]
    if net_focus:
        pool += [
            {
                "title": "查看监听端口",
                "summary": "列出正在监听的 TCP 端口和进程。",
                "cmd": "lsof -nP -iTCP -sTCP:LISTEN",
                "risk": "low",
            },
            {
                "title": "查看活跃 TCP 连接数",
                "summary": "统计已建立的 TCP 连接数，辅助判断是否存在连接风暴。",
                "cmd": "netstat -an -p tcp | grep ESTABLISHED | wc -l",
                "risk": "medium",
            },
        ]
    if disk_focus:
        pool += [
            {
                "title": "查看磁盘使用率",
                "summary": "读取各挂载点的使用率。",
                "cmd": "df -h",
                "risk": "low",
            },
            {
                "title": "查看磁盘 I/O",
                "summary": "一次采样磁盘传输速率，辅助判断是否 I/O 瓶颈。",
                "cmd": "iostat -d -w 1 -c 2",
                "risk": "medium",
            },
        ]

    # ---- 中优先级：通用深入排查（每轮都可以往前推进）----
    pool += [
        {
            "title": "按 CPU 排序前 20 进程",
            "summary": "按 CPU 倒序补充一轮进程快照。",
            "cmd": "ps -Ao pid,comm,%cpu,%mem,time -r | head -20",
            "risk": "low",
        },
        {
            "title": "按内存排序前 20 进程",
            "summary": "按内存倒序补充一轮进程快照。",
            "cmd": "ps -Ao pid,comm,%cpu,%mem,rss -m | head -20",
            "risk": "low",
        },
        {
            "title": "查看系统 uptime 与负载",
            "summary": "读取系统运行时间和 load average。",
            "cmd": "uptime",
            "risk": "low",
        },
        {
            "title": "查看虚拟内存状态",
            "summary": "读取 vm_stat 页面级内存统计。",
            "cmd": "vm_stat",
            "risk": "low",
        },
        {
            "title": "查看 swap 使用",
            "summary": "读取 swap 使用总量。",
            "cmd": "sysctl vm.swapusage",
            "risk": "low",
        },
        {
            "title": "查看磁盘使用率",
            "summary": "读取各挂载点使用率，确认是否存在磁盘空间瓶颈。",
            "cmd": "df -h",
            "risk": "low",
        },
        {
            "title": "查看监听端口",
            "summary": "列出正在监听的 TCP 端口和对应进程。",
            "cmd": "lsof -nP -iTCP -sTCP:LISTEN",
            "risk": "low",
        },
        {
            "title": "查看启动时间与基础信息",
            "summary": "读取 hostname / 系统版本。",
            "cmd": "sw_vers",
            "risk": "low",
        },
        {
            "title": "读取最近 1 分钟 error 日志",
            "summary": "排查最近系统 error 级别日志。",
            "cmd": 'log show --last 1m --predicate eventMessage CONTAINS[c] "error" --style compact',
            "risk": "medium",
        },
        {
            "title": "查看硬件核心数",
            "summary": "读取 CPU 核心数，辅助判断并发压力。",
            "cmd": "sysctl -n hw.ncpu hw.logicalcpu hw.physicalcpu",
            "risk": "low",
        },
    ]

    # 去重：按 cmd 文本去重，同时跳过已执行的命令
    unique: list[dict[str, str]] = []
    seen_commands: set[str] = set(executed_commands)
    for item in pool:
        cmd = item["cmd"]
        if cmd in seen_commands:
            continue
        seen_commands.add(cmd)
        # 归一化 risk 字段，缺省按 medium 保守处理（走审批）
        risk = str(item.get("risk") or "medium").strip().lower()
        if risk not in {"low", "medium", "high"}:
            risk = "medium"
        normalized = dict(item)
        normalized["risk"] = risk
        unique.append(normalized)
    return unique[:2]


def _format_iteration_approval_answer(
    *,
    iteration_count: int,
    iteration_limit: int,
    commands: list[dict[str, str]],
    hypotheses: list[dict],
    auto_executed: list[dict[str, str]] | None = None,
) -> str:
    lines = [
        f"当前已完成 {iteration_count} 轮自动排查，接下来准备执行第 {iteration_count + 1} 轮只读命令。",
        f"最大自动迭代轮数为 {iteration_limit} 轮，中/高风险命令需要你的审批后继续。",
    ]
    if auto_executed:
        lines.append("")
        lines.append("本轮已自动执行的低风险命令：")
        for idx, item in enumerate(auto_executed, start=1):
            lines.append(f"{idx}. {item.get('title', '')}")
            lines.append(f"   - 命令：`{item.get('cmd', '')}`  · 风险等级：low")
    lines.append("")
    lines.append("待审批执行的命令：")
    for index, item in enumerate(commands, start=1):
        risk = str(item.get("risk") or "medium").lower()
        lines.append(f"{index}. {item['title']}  · 风险等级：{risk}")
        lines.append(f"   - 目的：{item['summary']}")
        lines.append(f"   - 命令：`{item['cmd']}`")
    if hypotheses:
        lines.append("")
        lines.append("当前主要怀疑：")
        for hypothesis in hypotheses[:3]:
            if not isinstance(hypothesis, dict):
                continue
            lines.append(f"- {hypothesis.get('description', '未提供描述')}")
    return "\n".join(lines)


def _format_local_health_answer(
    *,
    summary: dict,
    hypotheses: list[dict],
    recommendations: list[dict],
) -> str:
    snap = summary.get("snapshot", {}) if isinstance(summary, dict) else {}
    lines: list[str] = []
    lines.append("现状摘要")
    lines.append(
        f"- CPU: {snap.get('cpu_percent')}%  load(1m): {snap.get('load_1m')}  freq(MHz): {snap.get('cpu_freq_mhz')}"
    )
    lines.append(
        f"- 内存: {snap.get('mem_percent')}%  swap_used_bytes: {snap.get('swap_used_bytes')}"
    )
    lines.append(
        f"- 热/功耗: thermal_pressure={snap.get('thermal_pressure')}  cpu_temp_c={snap.get('cpu_temp_c')}"
    )
    lines.append("")
    lines.append("最可能根因 Top3")
    if not hypotheses:
        lines.append("- (暂无高置信度根因假设；建议先做短时序二次采样)")
    else:
        for idx, h in enumerate(hypotheses[:3], start=1):
            lines.append(
                f"- #{idx} ({h.get('confidence', 0):.2f}) {h.get('description','')}"
                + (f"  evidence={h.get('evidence_refs', [])}" if h.get("evidence_refs") else "")
            )
    lines.append("")
    lines.append("建议清单（不自动执行）")
    if not recommendations:
        lines.append("- P0: 先执行一次 `ops.macos.timeseries_probe`（60s）确认是否持续异常")
    else:
        for r in recommendations:
            prio = r.get("priority", "P2")
            action = r.get("action", "")
            rationale = r.get("rationale", "")
            lines.append(f"- {prio}: {action} | {rationale}")
            cmds = r.get("suggested_commands", []) or []
            for cmd in cmds[:3]:
                lines.append(f"  command: {cmd}")
    return "\n".join(lines)


def _detect_anomalies(query: str) -> list[str]:
    q = query.lower()
    anomalies: list[str] = []
    if "cpu" in q:
        anomalies.append("high cpu")
    if "memory" in q or "oom" in q:
        anomalies.append("high memory")
    if "disk" in q or "io" in q:
        anomalies.append("disk pressure")
    if not anomalies:
        anomalies = ["application error", "dependency health", "resource baseline"]
    return anomalies[:3]


def _summarize_worker_budgets(worker_results: list[dict]) -> list[dict]:
    return [
        {
            "worker_id": item.get("worker_id", ""),
            "tool_calls_used": item.get("tool_calls_used", 0),
            "max_tool_calls": item.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS),
            "elapsed_seconds": item.get("elapsed_seconds", 0),
            "max_execution_seconds": item.get(
                "max_execution_seconds", DEFAULT_MAX_EXECUTION_SECONDS
            ),
            "budget_exhausted": item.get("budget_exhausted", False),
            "budget_exhausted_reason": item.get("budget_exhausted_reason", ""),
        }
        for item in worker_results
    ]


def _flatten_worker_records(worker_results: list[dict], field: str) -> list[dict]:
    flattened: list[dict] = []
    for worker in worker_results:
        worker_id = str(worker.get("worker_id", "") or "")
        for item in worker.get(field, []) or []:
            if isinstance(item, dict):
                flattened.append({"worker_id": worker_id, **item})
            else:
                flattened.append({"worker_id": worker_id, "value": item})
    return flattened


def _build_remediation_plan(
    *,
    anomaly_list: list[str],
    execution_target: str,
    target_host: str,
    evidence: list[dict],
) -> list[str]:
    # Keep the plan conservative: summarize what evidence was collected and propose safe next checks.
    sources = []
    for item in evidence:
        status = str(item.get("status", "") or "")
        if status == "failed":
            continue
        sources.append(str(item.get("source_type", "") or ""))
    source_summary = ", ".join(sorted({s for s in sources if s})) or "no trusted evidence"

    target_label = execution_target
    if execution_target == "remote" and target_host:
        target_label = f"remote({target_host})"

    plan = [
        f"review collected evidence ({source_summary}) on {target_label}",
        "collect latest metrics for the anomaly window",
        "inspect top processes and recent errors based on the snapshots",
    ]
    # Only include high-risk remediation suggestions when the anomalies indicate potential service impact.
    if any("error" in a or "dependency" in a for a in anomaly_list):
        plan.append("confirm whether restart or config rollback is required")
    return plan
