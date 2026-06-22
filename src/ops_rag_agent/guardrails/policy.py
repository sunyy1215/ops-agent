from __future__ import annotations

import re
from typing import Any, Iterable

PROMPT_INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "reveal system prompt",
    "show hidden prompt",
    "bypass guardrails",
    "act as system",
)
SENSITIVE_DATA_PATTERNS: tuple[str, ...] = (
    "api key",
    "access token",
    "private key",
    "password",
    "ssh key",
)
HIGH_RISK_ACTION_PATTERNS: tuple[str, ...] = (
    "restart",
    "rollback",
    "delete",
    "drop",
    "truncate",
    "shutdown",
    "sudo",
    "chmod",
    "chown",
    "rm ",
)
PRIVILEGED_COMMAND_PREFIXES: tuple[str, ...] = (
    "sudo ",
    "rm ",
    "mv ",
    "chmod ",
    "chown ",
    "launchctl ",
    "systemctl ",
    "service ",
)


def allow_guardrail_result() -> dict[str, Any]:
    return {"status": "passed", "action": "allow", "events": []}


def review_input_guardrails(query: str, *, enable_model_review: bool = True) -> dict[str, Any]:
    normalized = query.lower()
    events: list[dict[str, Any]] = []

    matched_injection = [item for item in PROMPT_INJECTION_PATTERNS if item in normalized]
    if matched_injection:
        events.append(
            {
                "layer": "rule",
                "status": "blocked",
                "action": "block",
                "reason": "prompt_injection_detected",
                "details": {"matches": matched_injection},
            }
        )

    matched_sensitive = [item for item in SENSITIVE_DATA_PATTERNS if item in normalized]
    if matched_sensitive:
        events.append(
            {
                "layer": "rule",
                "status": "blocked",
                "action": "block",
                "reason": "sensitive_data_request_detected",
                "details": {"matches": matched_sensitive},
            }
        )

    # Minimal "model-like" review: a heuristic hook that can be replaced by an LLM judge.
    if enable_model_review and _looks_like_policy_evasion(normalized):
        events.append(
            {
                "layer": "model",
                "status": "blocked",
                "action": "block",
                "reason": "semantic_policy_evasion_detected",
                "details": {"query_excerpt": query[:160]},
            }
        )

    return _finalize_guardrail_events(events)


def review_agent_guardrails(
    *,
    route: str,
    state: dict[str, Any],
    updates: dict[str, Any],
    skill_registry: Any | None = None,
    allowed_business_domains: Iterable[str] | None = None,
    enable_model_review: bool = True,
) -> dict[str, Any]:
    del state
    allowed_domains = set(allowed_business_domains or [])
    events: list[dict[str, Any]] = []

    remediation_plan = [str(step) for step in updates.get("remediation_plan", [])]
    risky_steps = [step for step in remediation_plan if _contains_high_risk_action(step)]
    if risky_steps:
        action = "require_approval" if route == "ops" else "block"
        status = "warn" if route == "ops" else "blocked"
        events.append(
            {
                "layer": "rule",
                "status": status,
                "action": action,
                "reason": "high_risk_remediation_detected",
                "details": {"steps": risky_steps},
            }
        )

    for call in updates.get("planned_skill_calls", []):
        skill_id = str(call.get("skill_id", "")).strip()
        if not skill_id:
            events.append(
                {
                    "layer": "tool",
                    "status": "blocked",
                    "action": "block",
                    "reason": "missing_skill_id",
                    "details": {"call": call},
                }
            )
            continue

        skill = None
        if skill_registry is not None:
            try:
                skill = skill_registry.get(skill_id)
            except KeyError:
                events.append(
                    {
                        "layer": "tool",
                        "status": "blocked",
                        "action": "block",
                        "reason": "unknown_skill_id",
                        "details": {"skill_id": skill_id},
                    }
                )
                continue

        if skill is not None:
            skill_domain = skill.spec.business_domain
            declared_domain = str(call.get("business_domain", skill_domain)).strip() or skill_domain
            if declared_domain != skill_domain:
                events.append(
                    {
                        "layer": "tool",
                        "status": "blocked",
                        "action": "block",
                        "reason": "skill_domain_mismatch",
                        "details": {
                            "skill_id": skill_id,
                            "declared_domain": declared_domain,
                            "registered_domain": skill_domain,
                        },
                    }
                )
            if allowed_domains and skill_domain not in allowed_domains:
                events.append(
                    {
                        "layer": "tool",
                        "status": "blocked",
                        "action": "block",
                        "reason": "skill_domain_not_allowed",
                        "details": {"skill_id": skill_id, "skill_domain": skill_domain},
                    }
                )

            requires_approval = bool(call.get("requires_approval", skill.spec.requires_approval))
            if requires_approval and not updates.get("approval_required", False):
                action = "require_approval" if route == "ops" else "block"
                status = "warn" if route == "ops" else "blocked"
                events.append(
                    {
                        "layer": "tool",
                        "status": status,
                        "action": action,
                        "reason": "approval_required_skill_detected",
                        "details": {"skill_id": skill_id},
                    }
                )

        cmd = str(call.get("arguments", {}).get("cmd", "")).strip().lower()
        if cmd and _looks_privileged_command(cmd):
            action = "require_approval" if route == "ops" else "block"
            status = "warn" if route == "ops" else "blocked"
            events.append(
                {
                    "layer": "tool",
                    "status": status,
                    "action": action,
                    "reason": "privileged_command_detected",
                    "details": {"skill_id": skill_id, "cmd": cmd},
                }
            )

    # RAG grounding nudge: warn when citations are missing.
    if route == "rag" and not updates.get("citations"):
        events.append(
            {
                "layer": "model",
                "status": "warn",
                "action": "allow",
                "reason": "missing_citations",
                "details": {"citation_count": 0},
            }
        )

    if enable_model_review and route == "ops" and risky_steps and not updates.get("approval_required", False):
        events.append(
            {
                "layer": "model",
                "status": "warn",
                "action": "require_approval",
                "reason": "semantic_high_risk_plan_requires_approval",
                "details": {"steps": risky_steps},
            }
        )

    return _finalize_guardrail_events(events)


def merge_guardrail_state(
    state: dict[str, Any],
    result: dict[str, Any],
    *,
    approval_payload: dict[str, Any] | None = None,
    block_message: str | None = None,
) -> dict[str, Any]:
    merged_events = [*state.get("guardrail_events", []), *result.get("events", [])]
    aggregate = _finalize_guardrail_events(merged_events)
    updates: dict[str, Any] = {
        "guardrail_events": merged_events,
        "guardrail_status": aggregate["status"],
        "guardrail_action": aggregate["action"],
    }

    if result.get("action") == "require_approval":
        payload = dict(approval_payload or state.get("approval_payload", {}))
        payload.setdefault("reason", "guardrails escalated approval")
        payload["guardrail_events"] = result.get("events", [])
        updates.update(
            {
                "approval_required": True,
                "approval_status": "pending",
                "workflow_status": "waiting_approval",
                "resumable_from": "approval_gate",
                "approval_payload": payload,
            }
        )

    if result.get("action") == "block":
        updates.update(
            {
                "workflow_status": "rejected",
                "approval_status": "rejected",
                "final_answer": block_message or "Request blocked by guardrails.",
            }
        )

    return updates


def _finalize_guardrail_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    action = "allow"
    status = "passed"
    if any(item.get("action") == "block" for item in events):
        action = "block"
    elif any(item.get("action") == "require_approval" for item in events):
        action = "require_approval"

    if any(item.get("status") == "blocked" for item in events):
        status = "blocked"
    elif any(item.get("status") == "warn" for item in events):
        status = "warn"

    return {"status": status, "action": action, "events": events}


def _contains_high_risk_action(text: str) -> bool:
    normalized = text.lower()
    return any(token in normalized for token in HIGH_RISK_ACTION_PATTERNS)


def _looks_privileged_command(command: str) -> bool:
    normalized = re.sub(r"\s+", " ", command.strip().lower())
    return any(normalized.startswith(prefix) for prefix in PRIVILEGED_COMMAND_PREFIXES)


def _looks_like_policy_evasion(text: str) -> bool:
    return bool(
        "ignore" in text
        and (
            "policy" in text
            or "guardrail" in text
            or "system prompt" in text
            or "instruction" in text
        )
    )
