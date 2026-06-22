from __future__ import annotations

import json

import pytest

from ops_rag_agent.ops.diagnostics import local_health_llm


def test_llm_analyze_fallback_on_non_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeMsg:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeLlm:
        def invoke(self, prompt: str):  # noqa: ANN001
            del prompt
            return _FakeMsg("not json")

    monkeypatch.setattr(local_health_llm, "build_reasoning_llm", lambda: _FakeLlm())
    res = local_health_llm.llm_analyze_local_health(evidence={"metrics": {"cpu": {"percent_total": 90}}})
    assert res.ok is False
    assert "missing_json_object" in res.error


def test_llm_recommend_sanitizes_dangerous_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeMsg:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeLlm:
        def invoke(self, prompt: str):  # noqa: ANN001
            del prompt
            payload = {
                "recommendations": [
                    {
                        "priority": "P0",
                        "action": "Restart service",
                        "rationale": "Try restart",
                        "evidence_refs": ["metrics.cpu.percent_total"],
                        "suggested_commands": ["systemctl restart nginx"],
                    }
                ]
            }
            return _FakeMsg(json.dumps(payload))

    monkeypatch.setattr(local_health_llm, "build_reasoning_llm", lambda: _FakeLlm())
    res = local_health_llm.llm_recommend_local_health(evidence={}, hypotheses=[])
    assert res.ok is True
    rec = res.value["recommendations"][0]
    assert "Risk note" in rec["rationale"]


def test_llm_summarize_local_health_returns_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeMsg:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeLlm:
        def invoke(self, prompt: str):  # noqa: ANN001
            assert "INPUT_JSON" in prompt
            return _FakeMsg("现状摘要\n- CPU 正常\n")

    monkeypatch.setattr(local_health_llm, "build_chat_llm", lambda: _FakeLlm())
    res = local_health_llm.llm_summarize_local_health(
        summary={"snapshot": {"cpu_percent": 20}},
        hypotheses=[],
        recommendations=[],
    )
    assert res.ok is True
    assert "现状摘要" in res.value["answer"]
