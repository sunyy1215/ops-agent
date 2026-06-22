"""端到端冒烟：skill_router + LangGraph 组合跑通。

用 stub LLM 注入 skill_router_node，绕过真实 OpenAI。验证：
  1) 用户 query → route → skill_router 低危 skill → finalize
  2) 高危 skill 触发 approval_gate interrupt；resume 后继续 finalize
"""

from __future__ import annotations

import json

import pytest

from ops_rag_agent.agents import skill_router_node
from ops_rag_agent.graph.app import build_graph
from ops_rag_agent.skills.base import Skill, SkillKind, SkillSpec
from ops_rag_agent.skills.registry import SkillRegistry


class _FakeLowRiskSkill(Skill):
    spec = SkillSpec(
        skill_id="ops.stub.low",
        name="stub-low",
        description="stub low-risk skill",
        version="1.0.0",
        business_domain="ops",
        kind=SkillKind.REGULAR,
        requires_approval=False,
        is_readonly=True,
        timeout_s=5,
        tags=("stub",),
        when_to_use="end-to-end smoke test",
        argument_schema={"type": "object"},
        example_invocations=(),
        risk_level="low",
    )

    def invoke(self, arguments):  # type: ignore[override]
        return "stub-low-output"


class _FakeHighRiskSkill(Skill):
    spec = SkillSpec(
        skill_id="ops.stub.high",
        name="stub-high",
        description="stub high-risk skill",
        version="1.0.0",
        business_domain="ops",
        kind=SkillKind.REGULAR,
        requires_approval=False,
        is_readonly=False,
        timeout_s=5,
        tags=("stub",),
        when_to_use="end-to-end smoke test for approval",
        argument_schema={"type": "object"},
        example_invocations=(),
        risk_level="high",
    )

    def invoke(self, arguments):  # type: ignore[override]
        return "stub-high-output"


def _scripted_llm(responses):
    queue = list(responses)

    def _invoke(_prompt: str) -> str:
        # 队列空时返回降级响应：让 intent/plan 走 heuristic 兜底，router 自行 finalize
        if not queue:
            return "{}"
        return queue.pop(0)

    return _invoke


# intent_analyzer / planner 都会先消费 LLM 响应，e2e 测试用占位 JSON 兜底，
# 真正断言对象是 router 的两个核心决策。
_INTENT_STUB = json.dumps(
    {
        "summary": "stub intent",
        "sub_questions": [],
        "domain_hints": ["ops_local"],
        "complexity": "moderate",
        "need_tools": True,
        "reasoning": "stub",
    }
)
_PLAN_STUB = json.dumps(
    {
        "reasoning": "stub plan",
        "steps": [
            {
                "title": "调用 stub skill",
                "intent": "stub",
                "suggested_skill_id": "ops.stub.low",
                "suggested_arguments": {},
                "expected_output": "stub-output",
            }
        ],
    }
)


def test_e2e_router_finalize_low_risk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Query → skill_router 调 low-risk skill → finalize → graph 结束。"""

    # 1) 替换 registry 内容：只塞一个低危 stub skill
    fake_registry = SkillRegistry()
    fake_registry.register(_FakeLowRiskSkill())

    # 2) 强制 build_skill_registry 返回假注册表
    import ops_rag_agent.graph.app as graph_app

    graph_app.build_graph.cache_clear()
    monkeypatch.setattr(graph_app, "build_skill_registry", lambda: fake_registry)

    # 3) 注入 stub LLM（顺序：intent → plan → router call → router finalize）
    script = [
        _INTENT_STUB,
        _PLAN_STUB,
        json.dumps(
            {
                "thought": "先跑 low",
                "action": "call_skill",
                "skill_id": "ops.stub.low",
                "arguments": {},
            }
        ),
        json.dumps(
            {
                "thought": "证据已拿到",
                "action": "finalize",
                "stop_reason": "enough_evidence",
                "final_answer": "最终结论：stub 通过",
            }
        ),
    ]
    shared_llm = _scripted_llm(script)
    # intent_analyzer / planner 都用 graph_app._default_llm_invoke
    monkeypatch.setattr(graph_app, "_default_llm_invoke", shared_llm)
    # skill_router 用自己的 _default_llm_invoke
    monkeypatch.setattr(skill_router_node, "_default_llm_invoke", shared_llm)

    # 4) 跑图。路由需要命中 ops，用运维关键词
    g = graph_app.build_graph()
    result = g.invoke(
        {"user_query": "帮我查一下本机 CPU 占用"},
        config={"configurable": {"thread_id": "e2e-low"}},
    )

    assert "stub 通过" in result["final_answer"]
    scratchpad = result.get("router_scratchpad") or []
    assert any(e.get("skill_id") == "ops.stub.low" for e in scratchpad)
    assert any(e.get("action") == "finalize" for e in scratchpad)
    assert any(
        item.get("skill_id") == "ops.stub.low"
        for item in (result.get("router_evidence", {}).get("skill_outputs") or [])
    )
    assert any(
        item.get("status") == "success"
        for item in (result.get("router_runtime_observations") or [])
    )
    assert any(
        item.get("status") == "skipped"
        for item in (result.get("router_validation_results") or [])
    )


def test_e2e_router_high_risk_triggers_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Query → skill_router 选 high-risk skill → 图 interrupt；resume → finalize。"""

    fake_registry = SkillRegistry()
    fake_registry.register(_FakeHighRiskSkill())

    import ops_rag_agent.graph.app as graph_app

    graph_app.build_graph.cache_clear()
    monkeypatch.setattr(graph_app, "build_skill_registry", lambda: fake_registry)

    # 第一轮选 high-risk；审批通过后 router 会重新跑，这次应直接 finalize
    _HIGH_PLAN_STUB = json.dumps(
        {
            "reasoning": "stub plan high",
            "steps": [
                {
                    "title": "调用 high skill",
                    "intent": "stub high",
                    "suggested_skill_id": "ops.stub.high",
                    "suggested_arguments": {"x": 1},
                    "expected_output": "stub-high-output",
                }
            ],
        }
    )
    script = [
        _INTENT_STUB,
        _HIGH_PLAN_STUB,
        json.dumps(
            {
                "thought": "需要执行 high",
                "action": "call_skill",
                "skill_id": "ops.stub.high",
                "arguments": {"x": 1},
            }
        ),
        json.dumps(
            {
                "thought": "已获得证据，收束",
                "action": "finalize",
                "stop_reason": "enough_evidence",
                "final_answer": "高危审批通过后顺利收束",
            }
        ),
    ]
    shared_llm = _scripted_llm(script)
    monkeypatch.setattr(graph_app, "_default_llm_invoke", shared_llm)
    monkeypatch.setattr(skill_router_node, "_default_llm_invoke", shared_llm)

    g = graph_app.build_graph()
    cfg = {"configurable": {"thread_id": "e2e-high"}}

    first = g.invoke({"user_query": "帮我排查端口占用"}, config=cfg)

    # 应停在 interrupt，等待审批
    interrupts = first.get("__interrupt__") or []
    # langgraph 在 `invoke` 返回值里把 interrupt 放在特殊字段；也可能通过 state 暴露
    # 用 get_state 再确认一遍
    snap = g.get_state(cfg)
    assert snap.values.get("approval_required") is True
    payload = snap.values.get("approval_payload", {})
    assert payload.get("type") == "skill_invocation"
    assert payload.get("skill_id") == "ops.stub.high"
    assert snap.values.get("router_runtime_observations", [])[0]["status"] == "pending_approval"
    assert snap.values.get("runtime_audit_records", [])[0]["phase"] == "policy"

    # 2) resume 审批通过
    from langgraph.types import Command

    g.invoke(Command(resume="approved"), config=cfg)
    final_snap = g.get_state(cfg)
    assert "顺利收束" in (final_snap.values.get("final_answer") or "")
