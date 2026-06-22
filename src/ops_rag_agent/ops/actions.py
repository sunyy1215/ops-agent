from __future__ import annotations

import re
import time
from typing import Any, Literal, TypedDict

from ops_rag_agent.observability import build_audit_event, utc_now
from ops_rag_agent.skills.registry import SkillRegistry


ActionType = Literal[
    "restart_service",
    "fetch_logs",
    "check_port",
    "check_process",
]
ActionCategory = Literal["diagnostic", "remediation"]
ActionStatus = Literal["planned", "approved", "done", "failed", "skipped"]


class RemediationAction(TypedDict, total=False):
    action_id: str
    action_type: ActionType
    category: ActionCategory
    status: ActionStatus
    requires_approval: bool

    execution_target: Literal["local", "remote"]
    target_host: str
    command: str
    risk_level: Literal["low", "medium", "high"]
    rollback_hint: str

    service_name: str
    process_name: str
    port: int
    log_lines: int

    # Execution trace fields (filled by executor).
    skill_id: str
    arguments: dict[str, Any]
    duration_ms: int
    success: bool
    executed_at: str
    exit_code: int | None
    stdout: str
    stderr: str
    action_result: str
    result: str


def build_remediation_actions(
    *,
    query: str,
    execution_target: str,
    target_host: str,
    remediation_plan: list[str],
) -> list[RemediationAction]:
    """
    Build a small, deterministic set of suggested actions.

    - Diagnostic actions are generally safe, but are still executed only after approval
      if the workflow is in the approval path.
    - Remediation actions must require approval by spec.
    """

    service_name = _infer_service_name(query)
    process_name = _infer_process_name(query) or service_name
    port = _infer_port(query)
    should_offer_restart = any("restart" in step.lower() for step in remediation_plan)

    actions: list[RemediationAction] = [
        {
            "action_id": "act-check-process",
            "action_type": "check_process",
            "category": "diagnostic",
            "status": "planned",
            "requires_approval": False,
            "execution_target": "remote" if execution_target == "remote" else "local",
            "target_host": target_host or "",
            "process_name": process_name,
        },
        {
            "action_id": "act-fetch-logs",
            "action_type": "fetch_logs",
            "category": "diagnostic",
            "status": "planned",
            "requires_approval": False,
            "execution_target": "remote" if execution_target == "remote" else "local",
            "target_host": target_host or "",
            "service_name": service_name,
            "log_lines": 200,
        },
    ]

    if port:
        actions.append(
            {
                "action_id": "act-check-port",
                "action_type": "check_port",
                "category": "diagnostic",
                "status": "planned",
                "requires_approval": False,
                "execution_target": "remote" if execution_target == "remote" else "local",
                "target_host": target_host or "",
                "port": int(port),
            }
        )

    if should_offer_restart:
        actions.append(
            {
                "action_id": "act-restart-service",
                "action_type": "restart_service",
                "category": "remediation",
                "status": "planned",
                "requires_approval": True,
                "execution_target": "remote" if execution_target == "remote" else "local",
                "target_host": target_host or "",
                "service_name": service_name,
            }
        )

    hydrated_actions: list[RemediationAction] = []
    for action in actions:
        hydrated_actions.append(_hydrate_action_defaults(action))
    return hydrated_actions


def actions_require_approval(actions: list[RemediationAction]) -> bool:
    return any(bool(a.get("requires_approval")) for a in actions)


def execute_approved_actions(state: dict[str, Any], skill_registry: SkillRegistry) -> dict[str, Any]:
    """
    Execute actions after approval is granted.

    This function is intentionally conservative:
    - It runs only when state.approval_status == "approved".
    - It uses only `ops.terminal.exec` (local) or `ops.remote.exec` (remote) as the executor.
    - It records per-action results back into approval_payload["actions"] and state.execution_results.
    """

    approval_status = str(state.get("approval_status", "")).strip().lower()
    if approval_status == "rejected":
        return {}

    payload = dict(state.get("approval_payload", {}) or {})
    actions: list[RemediationAction] = list(payload.get("actions", []) or [])
    if not actions:
        return {}

    executed_calls = list(state.get("executed_skill_calls", []) or [])
    planned_calls = list(state.get("planned_skill_calls", []) or [])
    execution_results: list[str] = list(state.get("execution_results", []) or [])
    audit_trail = list(state.get("audit_trail", []) or [])

    for idx, action in enumerate(actions):
        status = str(action.get("status", "planned"))
        if status in {"done", "failed", "skipped"}:
            continue
        if bool(action.get("requires_approval", False)) and approval_status != "approved":
            continue

        exec_target = str(action.get("execution_target") or state.get("ops_execution_target", {}).get("execution_target") or "local")
        target_host = str(action.get("target_host") or state.get("ops_execution_target", {}).get("target_host") or "")

        skill_id = "ops.remote.exec" if exec_target == "remote" else "ops.terminal.exec"
        cmd, postprocess = _build_command_and_postprocess(action)
        action["command"] = cmd
        arguments: dict[str, Any] = {"cmd": cmd, "approved": True}
        if exec_target == "remote":
            arguments["target_host"] = target_host

        start = time.monotonic()
        raw_result: Any = ""
        call_status: str = "done"
        try:
            skill = skill_registry.get(skill_id)
            raw_result = skill.invoke(arguments)
            result_text = str(raw_result)
            if result_text.startswith(("SECURITY_ERROR", "APPROVAL_REQUIRED", "ERROR")):
                call_status = "failed"
        except Exception as exc:  # noqa: BLE001 - safety net; keep workflow running
            call_status = "failed"
            raw_result = f"ERROR: {type(exc).__name__}: {exc}"
            result_text = str(raw_result)
        duration_ms = int((time.monotonic() - start) * 1000)

        # Apply optional post-processing for readability (no extra remote commands).
        if postprocess is not None:
            try:
                result_text = postprocess(result_text)
            except Exception:
                # Never fail the workflow because of formatting.
                pass

        success = call_status == "done"
        parsed_result = _parse_exec_result(raw_result)
        executed_at = utc_now()

        action["skill_id"] = skill_id
        action["arguments"] = dict(arguments)
        action["duration_ms"] = duration_ms
        action["success"] = success
        action["executed_at"] = executed_at
        action["exit_code"] = parsed_result["exit_code"]
        action["stdout"] = parsed_result["stdout"]
        action["stderr"] = parsed_result["stderr"]
        action["action_result"] = result_text
        action["result"] = result_text
        action["status"] = "done" if success else "failed"

        # Track skill calls in state for guardrails/auditing parity.
        planned_calls.append(
            {
                "skill_id": skill_id,
                "arguments": dict(arguments),
                "version": "",
                "business_domain": "ops",
                "kind": "regular",
                "requires_approval": bool(action.get("requires_approval", False)),
                "status": "approved",
            }
        )
        executed_calls.append(
            {
                "skill_id": skill_id,
                "arguments": dict(arguments),
                "version": "",
                "business_domain": "ops",
                "kind": "regular",
                "requires_approval": bool(action.get("requires_approval", False)),
                "status": "done" if success else "failed",
                "result": result_text,
            }
        )

        audit_trail.append(
            build_audit_event(
                "skill_executed",
                node="execute_actions",
                details={
                    "skill_id": skill_id,
                    "target_host": target_host if exec_target == "remote" else "",
                    "arguments": dict(arguments),
                    "duration_ms": duration_ms,
                    "success": success,
                    "action_id": action.get("action_id", f"action-{idx}"),
                    "action_type": action.get("action_type", ""),
                    "risk_level": action.get("risk_level", ""),
                },
            )
        )

        execution_results.append(
            f"action_id={action.get('action_id','')}\n"
            f"action_type={action.get('action_type','')}\n"
            f"skill_id={skill_id}\n"
            f"status={action['status']}\n"
            f"executed_at={executed_at}\n"
            f"{result_text}"
        )

    payload["actions"] = actions
    return {
        "approval_payload": payload,
        "execution_results": execution_results,
        "planned_skill_calls": planned_calls,
        "executed_skill_calls": executed_calls,
        "audit_trail": audit_trail,
    }


def _infer_service_name(query: str) -> str:
    q = str(query or "")
    lowered = q.lower()
    for token in ("nginx", "redis", "mysql", "postgres", "kafka"):
        if token in lowered:
            return token
    # Try patterns like "service xxx"
    m = re.search(r"\bservice\s+([A-Za-z0-9_.-]{2,64})\b", lowered)
    if m:
        return m.group(1)
    return "app"


def _infer_process_name(query: str) -> str:
    lowered = str(query or "").lower()
    m = re.search(r"\bprocess\s+([A-Za-z0-9_.-]{2,64})\b", lowered)
    if m:
        return m.group(1)
    return ""


def _infer_port(query: str) -> int:
    text = str(query or "")
    # "port 8080" / "端口 8080"
    m = re.search(r"(?:\bport\b|端口)\s*[:=]?\s*(\d{2,5})", text, flags=re.IGNORECASE)
    if not m:
        # ":8080"
        m = re.search(r":(\d{2,5})\b", text)
    if not m:
        return 0
    port = int(m.group(1))
    if 1 <= port <= 65535:
        return port
    return 0


def _build_command_and_postprocess(
    action: RemediationAction,
) -> tuple[str, Any | None]:
    action_type = str(action.get("action_type", "") or "")

    if action_type == "restart_service":
        service_name = str(action.get("service_name", "app")).strip() or "app"
        return (f"systemctl restart {service_name}", None)

    if action_type == "fetch_logs":
        service_name = str(action.get("service_name", "app")).strip() or "app"
        log_lines = int(action.get("log_lines", 200) or 200)
        log_lines = max(20, min(log_lines, 400))
        return (f"journalctl -u {service_name} -n {log_lines} --no-pager", None)

    if action_type == "check_port":
        port = int(action.get("port", 0) or 0)

        def postprocess(text: str) -> str:
            if not port:
                return text
            marker = "\nstdout:\n"
            idx = text.find(marker)
            if idx == -1:
                return text
            stdout = text[idx + len(marker) :]
            # Heuristic filter: show only lines that mention ":<port>".
            filtered = "\n".join(
                line for line in stdout.splitlines() if f":{port}" in line
            )
            return text[: idx + len(marker)] + (filtered or "<no_match>")

        return ("ss -lnt", postprocess)

    if action_type == "check_process":
        process_name = str(action.get("process_name", "")).strip()

        def postprocess(text: str) -> str:
            if not process_name:
                return text
            marker = "\nstdout:\n"
            idx = text.find(marker)
            if idx == -1:
                return text
            stdout = text[idx + len(marker) :]
            lines = stdout.splitlines()
            header = lines[:1]
            body = lines[1:]
            matches = [
                line for line in body if process_name.lower() in line.lower()
            ][:50]
            clipped = "\n".join([*header, *(matches or ["<no_match>"])])
            return text[: idx + len(marker)] + clipped

        # Avoid pipes: filter locally.
        return ("ps -eo pid,comm,args", postprocess)

    # Unknown action: keep it safe.
    return ("date", None)


def _hydrate_action_defaults(action: RemediationAction) -> RemediationAction:
    cmd, _ = _build_command_and_postprocess(action)
    action_type = str(action.get("action_type", "") or "")
    risk_level = "low"
    rollback_hint = "not_required"
    if action_type == "restart_service":
        risk_level = "high"
        rollback_hint = "restart the service again only if safe, otherwise roll back the last config or deployment change"
    elif action_type == "fetch_logs":
        risk_level = "low"
    elif action_type == "check_port":
        risk_level = "low"
    elif action_type == "check_process":
        risk_level = "low"

    hydrated = dict(action)
    hydrated["command"] = cmd
    hydrated["risk_level"] = risk_level
    hydrated["rollback_hint"] = rollback_hint
    return hydrated


def _parse_exec_result(result_text: Any) -> dict[str, Any]:
    if isinstance(result_text, dict):
        return {
            "exit_code": result_text.get("exit_code"),
            "stdout": str(result_text.get("stdout") or ""),
            "stderr": str(result_text.get("stderr") or ""),
        }

    text = str(result_text or "")
    exit_code: int | None = None
    stdout = ""
    stderr = ""

    exit_match = re.search(r"(^|\n)exit_code=(-?\d+)", text)
    if exit_match:
        exit_code = int(exit_match.group(2))

    stdout_match = re.search(r"\nstdout:\n(?P<body>.*?)(?:\nstderr:\n|\Z)", text, flags=re.DOTALL)
    if stdout_match:
        stdout = stdout_match.group("body")

    stderr_match = re.search(r"\nstderr:\n(?P<body>.*)\Z", text, flags=re.DOTALL)
    if stderr_match:
        stderr = stderr_match.group("body")

    return {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
    }
