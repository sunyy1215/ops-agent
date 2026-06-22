from __future__ import annotations

from copy import deepcopy
import re
import time
from typing import Any

from ops_rag_agent.config import settings
from ops_rag_agent.skills.registry import SkillRegistry

DEFAULT_MAX_TOOL_CALLS = 15
DEFAULT_MAX_EXECUTION_SECONDS = 10 * 60
EXECUTION_TARGET_LOCAL = "local"
EXECUTION_TARGET_REMOTE = "remote"
SOURCE_QUALITY_PRIORITY = (
    "official_documentation",
    "monitoring_platform",
    "system_metrics",
    "internal_knowledge_base",
    "application_logs",
    "operator_notes",
    "community_forum",
)
SOURCE_QUALITY_RANK = {
    source: len(SOURCE_QUALITY_PRIORITY) - idx
    for idx, source in enumerate(SOURCE_QUALITY_PRIORITY)
}


def infer_execution_target_from_query(query: str) -> tuple[str, str]:
    """Infer whether to execute locally or remotely, and return (execution_target, target_host).

    Heuristics only; can be replaced by LLM-based planning later.
    """

    normalized = (query or "").strip()
    lowered = normalized.lower()
    if not normalized:
        return (EXECUTION_TARGET_LOCAL, "")

    # Explicit remote intent keywords.
    remote_markers = ("remote", "ssh", "远端", "远程", "线上", "生产", "prod", "node", "主机", "机器")
    explicit_remote = any(marker in lowered for marker in remote_markers)

    # Try to extract an explicit target host from the query.
    # Accept "user@host", IPv4, or a simple hostname.
    candidates: list[str] = []
    candidates.extend(re.findall(r"\b[\w.\-]+@[\w.\-]+\b", normalized))
    candidates.extend(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", normalized))
    candidates.extend(re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9.\-]{2,62}\b", normalized))

    allowed_hosts = set(settings.allowed_remote_ssh_hosts)
    target_host = ""
    for cand in candidates:
        if cand in allowed_hosts:
            target_host = cand
            break

    if explicit_remote or target_host:
        return (
            EXECUTION_TARGET_REMOTE,
            target_host or settings.remote_ssh_default_host,
        )
    return (EXECUTION_TARGET_LOCAL, "")


def build_ops_worker_tasks(
    anomaly_list: list[str],
    *,
    execution_target: str = EXECUTION_TARGET_LOCAL,
    target_host: str = "",
) -> list[dict]:
    """Parent agent can create at most 3 child investigation tasks."""

    resolved_target = execution_target or EXECUTION_TARGET_LOCAL
    resolved_host = target_host or (settings.remote_ssh_default_host if resolved_target == EXECUTION_TARGET_REMOTE else "")

    tasks: list[dict[str, Any]] = []
    for idx, anomaly in enumerate(anomaly_list[: settings.max_ops_workers], start=1):
        tasks.append(
            {
                "worker_id": f"worker-{idx}",
                "title": f"Investigate anomaly #{idx}",
                "anomaly_type": anomaly,
                "execution_target": resolved_target,
                "target_host": resolved_host,
                "findings": [],
                "recommendation": "",
                "max_tool_calls": DEFAULT_MAX_TOOL_CALLS,
                "max_execution_seconds": DEFAULT_MAX_EXECUTION_SECONDS,
                "tool_calls_used": 0,
                "elapsed_seconds": 0,
                "budget_exhausted": False,
                "budget_exhausted_reason": "",
                "evidence": [],
                "skill_calls": [],
                "failed_skills": [],
                "deprioritized_sources": [],
                "source_priority": list(SOURCE_QUALITY_PRIORITY),
            }
        )
    return tasks


def build_source_quality_policy() -> dict[str, Any]:
    return {
        "preferred_sources": list(SOURCE_QUALITY_PRIORITY[:4]),
        "deprioritized_sources": list(SOURCE_QUALITY_PRIORITY[4:]),
        "policy": (
            "Prefer official documentation, monitoring platforms, system metrics, "
            "and internal knowledge bases before relying on lower-trust sources."
        ),
    }


def run_worker_task(worker_task: dict, skill_registry: SkillRegistry) -> dict:
    task = deepcopy(worker_task)
    anomaly = str(task["anomaly_type"])
    task.setdefault("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)
    task.setdefault("max_execution_seconds", DEFAULT_MAX_EXECUTION_SECONDS)
    task.setdefault("source_priority", list(SOURCE_QUALITY_PRIORITY))
    task.setdefault("execution_target", EXECUTION_TARGET_LOCAL)
    task.setdefault("target_host", "")
    task.setdefault("skill_calls", [])
    task.setdefault("failed_skills", [])

    ranked_sources = rank_sources_for_anomaly(
        anomaly,
        execution_target=str(task.get("execution_target") or EXECUTION_TARGET_LOCAL),
    )
    preferred_sources = [item for item in ranked_sources if item["preferred"]]
    deprioritized_sources = [item for item in ranked_sources if not item["preferred"]]

    findings: list[str] = []
    evidence: list[dict[str, Any]] = []
    skill_calls: list[dict[str, Any]] = []
    failed_skills: list[dict[str, Any]] = []
    tool_calls_used = 0
    elapsed_seconds = 0
    budget_exhausted = False
    budget_exhausted_reason = ""
    skipped_due_to_tool_budget = False
    skipped_due_to_time_budget = False

    for source in preferred_sources:
        next_tool_calls = tool_calls_used + int(source["tool_calls"])
        next_elapsed_seconds = elapsed_seconds + int(source["estimated_duration_s"])
        if next_tool_calls > task["max_tool_calls"]:
            skipped_due_to_tool_budget = True
            continue
        if next_elapsed_seconds > task["max_execution_seconds"]:
            skipped_due_to_time_budget = True
            continue

        tool_calls_used = next_tool_calls
        elapsed_seconds = next_elapsed_seconds
        skill_id = str(source.get("skill_id", "") or "").strip()
        call_record: dict[str, Any] | None = None
        result_text = ""
        status: str = "done"

        if skill_id:
            try:
                skill = skill_registry.get(skill_id)
                args = _build_skill_arguments(
                    skill_id,
                    anomaly=anomaly,
                    execution_target=str(task.get("execution_target")),
                    target_host=str(task.get("target_host")),
                )
                requires_approval = bool(skill.spec.requires_approval)
                call_record = {
                    "skill_id": skill_id,
                    "arguments": args,
                    "version": skill.spec.version,
                    "business_domain": skill.spec.business_domain,
                    "kind": skill.spec.kind.value,
                    "requires_approval": requires_approval,
                    "status": "approved" if not requires_approval else "planned",
                }

                if requires_approval:
                    status = "failed"
                    result_text = "APPROVAL_REQUIRED: skill requires approval"
                    call_record["duration_ms"] = 0
                    call_record["success"] = False
                    failed_skills.append(
                        {
                            "skill_id": skill_id,
                            "failure_type": "approval_required",
                            "reason": "skill_spec_requires_approval",
                        }
                    )
                else:
                    start = time.monotonic()
                    result_text = str(skill.invoke(args))
                    duration_ms = int((time.monotonic() - start) * 1000)
                    call_record["status"] = "done"
                    call_record["result"] = result_text
                    call_record["duration_ms"] = duration_ms
                    call_record["success"] = True
                    if result_text.startswith("SECURITY_ERROR"):
                        status = "failed"
                        call_record["status"] = "failed"
                        call_record["success"] = False
                        failed_skills.append(
                            {
                                "skill_id": skill_id,
                                "failure_type": "security_error",
                                "reason": result_text[:240],
                            }
                        )
                    elif result_text.startswith("APPROVAL_REQUIRED"):
                        status = "failed"
                        call_record["status"] = "failed"
                        call_record["success"] = False
                        failed_skills.append(
                            {
                                "skill_id": skill_id,
                                "failure_type": "approval_required",
                                "reason": result_text[:240],
                            }
                        )
            except Exception as exc:  # noqa: BLE001 - execution safety net
                status = "failed"
                result_text = f"ERROR: {type(exc).__name__}: {exc}"
                if call_record is None:
                    call_record = {
                        "skill_id": skill_id,
                        "arguments": {},
                        "version": "",
                        "business_domain": "ops",
                        "kind": "regular",
                        "requires_approval": False,
                        "status": "failed",
                    }
                call_record["status"] = "failed"
                call_record["result"] = result_text
                call_record["duration_ms"] = int(call_record.get("duration_ms", 0) or 0)
                call_record["success"] = False
                failed_skills.append(
                    {
                        "skill_id": skill_id,
                        "failure_type": "exception",
                        "reason": result_text[:240],
                    }
                )

        findings.append(
            source["finding"]
            if status != "failed"
            else f"{source['finding']} (skill failed: {skill_id or 'n/a'})"
        )
        if call_record is not None:
            skill_calls.append(call_record)
        evidence.append(
            {
                "source_type": source["source_type"],
                "summary": source["summary"],
                "quality_rank": source["quality_rank"],
                "tool_calls": int(source["tool_calls"]),
                "estimated_duration_s": int(source["estimated_duration_s"]),
                "skill_id": skill_id,
                "execution_target": task.get("execution_target", EXECUTION_TARGET_LOCAL),
                "target_host": task.get("target_host", ""),
                "status": status,
                # Keep a short excerpt to avoid exploding state size.
                "result_excerpt": result_text[:800] if result_text else "",
            }
        )

    if skipped_due_to_tool_budget or skipped_due_to_time_budget:
        budget_exhausted = True
        if skipped_due_to_time_budget:
            budget_exhausted_reason = "execution_time_budget_exceeded"
        else:
            budget_exhausted_reason = "tool_call_budget_exceeded"

    task["findings"] = findings or [
        (
            f"Budget prevented additional investigation for anomaly '{anomaly}'. "
            "Return current baseline and request manual follow-up."
        )
    ]
    task["recommendation"] = _build_recommendation(
        anomaly=anomaly,
        findings=task["findings"],
        evidence=evidence,
        budget_exhausted=budget_exhausted,
        budget_exhausted_reason=budget_exhausted_reason,
    )
    task["tool_calls_used"] = tool_calls_used
    task["elapsed_seconds"] = elapsed_seconds
    task["budget_exhausted"] = budget_exhausted
    task["budget_exhausted_reason"] = budget_exhausted_reason
    task["evidence"] = evidence
    task["skill_calls"] = skill_calls
    task["failed_skills"] = failed_skills
    task["deprioritized_sources"] = [
        {
            "source_type": item["source_type"],
            "summary": item["summary"],
            "quality_rank": item["quality_rank"],
        }
        for item in deprioritized_sources
    ]
    return task


def rank_sources_for_anomaly(anomaly: str, *, execution_target: str) -> list[dict[str, Any]]:
    sources = _build_source_candidates(anomaly, execution_target=execution_target)
    return sorted(
        sources,
        key=lambda item: (
            item["quality_rank"],
            -item["estimated_duration_s"],
        ),
        reverse=True,
    )


def _build_source_candidates(anomaly: str, *, execution_target: str) -> list[dict[str, Any]]:
    normalized = anomaly.lower()
    exec_target = execution_target or EXECUTION_TARGET_LOCAL
    system_skill_id = (
        "ops.remote.snapshot" if exec_target == EXECUTION_TARGET_REMOTE else "ops.local.snapshot"
    )
    baseline_sources: list[dict[str, Any]] = [
        {
            "source_type": "monitoring_platform",
            "summary": "Query current and recent metrics from Prometheus.",
            "finding": "Monitoring data highlights the current anomaly window and affected scope.",
            "tool_calls": 2,
            "estimated_duration_s": 45,
            "skill_id": "ops.prometheus.query",
        },
        {
            "source_type": "system_metrics",
            "summary": (
                "Collect remote host CPU, memory, disk, and hot process snapshot."
                if exec_target == EXECUTION_TARGET_REMOTE
                else "Collect local host CPU, memory, disk, and hot process snapshot."
            ),
            "finding": "System metrics confirm whether the issue is host-level or process-specific.",
            "tool_calls": 1,
            "estimated_duration_s": 20,
            "skill_id": system_skill_id,
        },
        {
            "source_type": "internal_knowledge_base",
            "summary": "Check prior incident notes and internal runbooks if available.",
            "finding": "Internal runbooks narrow down common mitigations for this anomaly family.",
            "tool_calls": 0,
            "estimated_duration_s": 30,
            "skill_id": "",
        },
        {
            "source_type": "community_forum",
            "summary": "Use ad-hoc community suggestions only if higher-trust evidence is insufficient.",
            "finding": "Community suggestions may provide hypotheses but require verification.",
            "tool_calls": 0,
            "estimated_duration_s": 15,
            "skill_id": "",
        },
    ]

    if "cpu" in normalized:
        baseline_sources[0]["finding"] = "Monitoring trends show whether the CPU spike is sustained or traffic-driven."
        baseline_sources[1]["finding"] = "Local snapshots identify hot processes, busy cores, and runaway executors."
    elif "memory" in normalized or "oom" in normalized:
        baseline_sources[0]["finding"] = "Monitoring trends show whether memory growth is sudden, periodic, or leak-like."
        baseline_sources[1]["finding"] = "Local snapshots expose resident memory pressure and top consumers."
    elif "disk" in normalized or "io" in normalized:
        baseline_sources[0]["finding"] = "Monitoring trends show IO saturation and disk pressure timing."
        baseline_sources[1]["finding"] = "Local snapshots expose filesystem utilization and heavy disk users."
    else:
        baseline_sources.append(
            {
                "source_type": "official_documentation",
                "summary": "Review official product or platform documentation for supported recovery steps.",
                "finding": "Official documentation helps validate the safest remediation path.",
                "tool_calls": 0,
                "estimated_duration_s": 40,
                "skill_id": "",
            }
        )

    for source in baseline_sources:
        source["quality_rank"] = SOURCE_QUALITY_RANK[source["source_type"]]
        source["preferred"] = source["source_type"] in SOURCE_QUALITY_PRIORITY[:4]
    return baseline_sources


def _build_recommendation(
    *,
    anomaly: str,
    findings: list[str],
    evidence: list[dict[str, Any]],
    budget_exhausted: bool,
    budget_exhausted_reason: str,
) -> str:
    if "cpu" in anomaly.lower():
        recommendation = "Inspect top CPU consumers, recent traffic changes, and deployment diffs."
    elif "memory" in anomaly.lower() or "oom" in anomaly.lower():
        recommendation = "Inspect memory growth, cache pressure, restart policy, and GC behavior."
    elif "disk" in anomaly.lower() or "io" in anomaly.lower():
        recommendation = "Inspect disk utilization, IO-heavy processes, and log growth."
    else:
        recommendation = "Collect metrics and runbook guidance before attempting remediation."

    source_types = ", ".join(item["source_type"] for item in evidence) or "no preferred evidence"
    summary = (
        f"{recommendation} Evidence priority respected: {source_types}. "
        f"Current findings: {len(findings)}."
    )
    if budget_exhausted:
        summary += (
            " Investigation stopped after reaching budget guardrail "
            f"({budget_exhausted_reason})."
        )
    return summary


def _build_skill_arguments(
    skill_id: str,
    *,
    anomaly: str,
    execution_target: str,
    target_host: str,
) -> dict[str, Any]:
    # Keep arguments deterministic and safe; real planner can refine later.
    if skill_id == "ops.prometheus.query":
        # A safe default query. Operators can refine to node/service-specific metrics later.
        return {"query": "up"}
    if skill_id == "ops.local.snapshot":
        return {"top_n": 8}
    if skill_id == "ops.remote.snapshot":
        return {"target_host": target_host, "top_n": 8}
    # Fallback: pass through a minimal context. Avoid injecting shell commands.
    return {"anomaly": anomaly, "execution_target": execution_target, "target_host": target_host}
