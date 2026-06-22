from __future__ import annotations

from typing import Any

import pytest

from ops_rag_agent.config import settings
from ops_rag_agent.ops.actions import build_remediation_actions, execute_approved_actions
from ops_rag_agent.skills.bootstrap import build_skill_registry


class _DummyCompleted:
    def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_build_remediation_actions_includes_restart_only_when_plan_mentions_restart() -> None:
    actions = build_remediation_actions(
        query="service nginx error",
        execution_target="remote",
        target_host="",
        remediation_plan=["collect metrics", "confirm whether restart is required"],
    )
    assert any(a.get("action_type") == "restart_service" for a in actions)
    restart_action = next(a for a in actions if a.get("action_type") == "restart_service")
    assert restart_action["command"] == "systemctl restart nginx"
    assert restart_action["risk_level"] == "high"
    assert "roll back" in restart_action["rollback_hint"]

    actions2 = build_remediation_actions(
        query="cpu alert",
        execution_target="local",
        target_host="",
        remediation_plan=["collect metrics", "inspect top processes"],
    )
    assert not any(a.get("action_type") == "restart_service" for a in actions2)
    assert all(a.get("command") for a in actions2)
    assert all(a.get("risk_level") == "low" for a in actions2)


def test_execute_approved_actions_executes_via_registry_and_backfills(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid real subprocess/ssh.
    import ops_rag_agent.skills.remote_ssh_exec as remote_ssh_exec
    import ops_rag_agent.skills.terminal_exec as terminal_exec

    def _fake_run(*args: Any, **kwargs: Any) -> _DummyCompleted:
        # Keep a small "stdout" body so post-processing paths are exercised.
        return _DummyCompleted(returncode=0, stdout="header\nline :80\nnginx\n", stderr="")

    monkeypatch.setattr(remote_ssh_exec.subprocess, "run", _fake_run)
    monkeypatch.setattr(terminal_exec.subprocess, "run", _fake_run)

    registry = build_skill_registry()
    target_host = settings.remote_ssh_default_host

    actions = build_remediation_actions(
        query="service nginx port 80 error",
        execution_target="remote",
        target_host=target_host,
        remediation_plan=["confirm whether restart is required"],
    )
    assert actions

    state = {
        "approval_status": "approved",
        "approval_payload": {"actions": actions},
        "ops_execution_target": {"execution_target": "remote", "target_host": target_host},
        "planned_skill_calls": [],
        "executed_skill_calls": [],
        "audit_trail": [],
    }

    updates = execute_approved_actions(state, registry)

    assert "approval_payload" in updates
    assert updates["approval_payload"]["actions"]
    assert all(a["status"] in {"done", "failed"} for a in updates["approval_payload"]["actions"])
    assert all("command" in a for a in updates["approval_payload"]["actions"])
    assert all("executed_at" in a for a in updates["approval_payload"]["actions"])
    assert all("action_result" in a for a in updates["approval_payload"]["actions"])
    assert all("stdout" in a and "stderr" in a for a in updates["approval_payload"]["actions"])
    assert updates.get("execution_results")
    assert updates.get("executed_skill_calls")

    # Audit fields required by spec.
    assert any(
        ev.get("event_type") == "skill_executed"
        and ev.get("details", {}).get("skill_id")
        and "duration_ms" in ev.get("details", {})
        and "success" in ev.get("details", {})
        for ev in updates.get("audit_trail", [])
    )


def test_execute_actions_runs_diagnostic_actions_without_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    import ops_rag_agent.skills.terminal_exec as terminal_exec

    def _fake_run(*args: Any, **kwargs: Any) -> _DummyCompleted:
        return _DummyCompleted(returncode=0, stdout="PID COMMAND ARGS\n123 app demo\n", stderr="")

    monkeypatch.setattr(terminal_exec.subprocess, "run", _fake_run)

    registry = build_skill_registry()
    actions = build_remediation_actions(
        query="cpu alert process app",
        execution_target="local",
        target_host="",
        remediation_plan=["inspect top processes"],
    )
    assert all(not a.get("requires_approval", False) for a in actions)

    updates = execute_approved_actions(
        {
            "approval_status": "not_required",
            "approval_payload": {"actions": actions},
            "ops_execution_target": {"execution_target": "local", "target_host": ""},
            "planned_skill_calls": [],
            "executed_skill_calls": [],
            "audit_trail": [],
        },
        registry,
    )

    assert updates["approval_payload"]["actions"]
    assert all(a["status"] == "done" for a in updates["approval_payload"]["actions"])
