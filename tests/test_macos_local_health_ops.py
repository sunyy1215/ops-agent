from __future__ import annotations

import json
from typing import Any

import pytest

from ops_rag_agent.agents.ops_agent import run_ops_agent
from ops_rag_agent.skills.bootstrap import build_skill_registry


class _DummyCompleted:
    def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_local_health_requires_powermetrics_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    import ops_rag_agent.skills.macos_gpu_profile as macos_gpu_profile

    # Avoid slow real system_profiler call.
    fake_profile = {"SPDisplaysDataType": [{"sppci_model": "Apple M", "spdisplays_vendor": "Apple"}]}
    monkeypatch.setattr(
        macos_gpu_profile.subprocess,
        "run",
        lambda *args, **kwargs: _DummyCompleted(stdout=json.dumps(fake_profile), stderr="", returncode=0),
    )

    registry = build_skill_registry()
    result = run_ops_agent(
        {
            "user_query": "温度很高，风扇很吵",
            "require_powermetrics_approval": True,
        },
        registry,
    )

    assert result["approval_required"] is True
    assert result["approval_payload"]["type"] == "collect_powermetrics"
    assert "powermetrics" in result["final_answer"].lower()


def test_local_health_runs_after_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    import ops_rag_agent.skills.macos_gpu_profile as macos_gpu_profile
    import ops_rag_agent.skills.macos_powermetrics as macos_powermetrics

    fake_profile = {"SPDisplaysDataType": [{"sppci_model": "Apple M", "spdisplays_vendor": "Apple"}]}
    monkeypatch.setattr(
        macos_gpu_profile.subprocess,
        "run",
        lambda *args, **kwargs: _DummyCompleted(stdout=json.dumps(fake_profile), stderr="", returncode=0),
    )

    # Provide a minimal powermetrics stdout that matches the parser patterns.
    powermetrics_stdout = (
        "CPU die temperature: 55.0 C\n"
        "Thermal Pressure: Nominal\n"
        "Fan: 1200 rpm\n"
        "Average frequency: 2400 MHz\n"
    )

    def _fake_run(argv: list[str], *args: Any, **kwargs: Any) -> _DummyCompleted:
        del argv
        return _DummyCompleted(stdout=powermetrics_stdout, stderr="", returncode=0)

    monkeypatch.setattr(macos_powermetrics.subprocess, "run", _fake_run)

    registry = build_skill_registry()
    result = run_ops_agent(
        {
            "user_query": "最近卡顿，怀疑热导致降频，想看 thermal pressure 和温度",
            "macos_powermetrics_authorized": True,
            "approval_status": "approved",
        },
        registry,
    )

    assert result["approval_required"] is False
    assert result.get("remediation_actions", []) == []
    assert "现状摘要" in result["final_answer"]
    assert result.get("ops_skill_calls")
    assert result.get("ops_evidence")
