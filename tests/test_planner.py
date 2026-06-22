"""单测：planner（执行计划生成器）。"""

from __future__ import annotations

from typing import Any

from ops_rag_agent.agents.intent_analyzer import IntentAnalysis
from ops_rag_agent.agents.planner import (
    ExecutionPlan,
    PlanStep,
    heuristic_plan,
    parse_plan,
    plan_execution,
)


class _FakeRegistry:
    """最小可用的 SkillRegistry 替身：提供 list_specs 接口。"""

    def __init__(self, specs: list[dict[str, Any]]) -> None:
        self._specs = specs

    def list_specs(
        self, *, allowed_business_domains: list[str] | None = None
    ) -> list[dict[str, Any]]:
        return list(self._specs)


_DUMMY_SPECS: list[dict[str, Any]] = [
    {
        "skill_id": "rag.search",
        "name": "rag.search",
        "description": "查内部文档/手册",
        "when_to_use": "用户问内部文档/手册时使用",
        "risk_level": "low",
        "requires_approval": False,
        "argument_schema": {"query": "str", "top_k": "int"},
        "example_invocations": [],
    },
    {
        "skill_id": "web.search",
        "name": "web.search",
        "description": "联网搜索最新信息",
        "when_to_use": "用户问最新/实时信息时使用",
        "risk_level": "low",
        "requires_approval": False,
        "argument_schema": {"query": "str"},
        "example_invocations": [],
    },
]


def test_parse_plan_extracts_steps() -> None:
    raw = """{
      "reasoning": "先查内部文档再总结",
      "steps": [
        {"title": "查 k8s 重启 playbook",
         "intent": "找内部排查指南",
         "suggested_skill_id": "rag.search",
         "suggested_arguments": {"query": "k8s pod 重启", "top_k": 5},
         "expected_output": "排查思路"}
      ]
    }"""
    plan = parse_plan(raw)
    assert isinstance(plan, ExecutionPlan)
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.suggested_skill_id == "rag.search"
    assert step.suggested_arguments["query"] == "k8s pod 重启"
    assert step.id.startswith("step-")
    assert step.status == "planned"


def test_parse_plan_handles_garbage() -> None:
    plan = parse_plan("not json")
    assert plan.steps == []
    assert plan.reasoning == "llm_output_not_json"


def test_plan_execution_always_runs_even_when_need_tools_false() -> None:
    """新策略：need_tools 已强制为 True，且 plan 永远先 rag.search。"""

    intent = IntentAnalysis(
        summary="hi", sub_questions=[], domain_hints=["knowledge_base"],
        complexity="simple", need_tools=True, reasoning="forced",
    )

    def llm_invoke(_prompt: str) -> str:
        # LLM 漏写 rag → planner 兜底注入
        return '{"reasoning":"forgot rag","steps":[]}'

    plan = plan_execution(
        intent=intent,
        registry=_FakeRegistry(_DUMMY_SPECS),
        llm_invoke=llm_invoke,
    )
    assert len(plan.steps) >= 1
    assert plan.steps[0].suggested_skill_id == "rag.search"


def test_plan_execution_invokes_llm_for_complex_query() -> None:
    intent = IntentAnalysis(
        summary="查 k8s 重启",
        sub_questions=["如何查 lastState"],
        domain_hints=["knowledge_base"],
        complexity="moderate",
        need_tools=True,
        reasoning="ok",
    )
    captured: dict[str, str] = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return (
            '{"reasoning":"先查 rag","steps":[{'
            '"title":"查 playbook","intent":"找内部资料",'
            '"suggested_skill_id":"rag.search",'
            '"suggested_arguments":{"query":"k8s 重启"},'
            '"expected_output":"排查思路"}]}'
        )

    plan = plan_execution(
        intent=intent,
        registry=_FakeRegistry(_DUMMY_SPECS),
        llm_invoke=fake_llm,
    )
    assert len(plan.steps) == 1
    assert plan.steps[0].suggested_skill_id == "rag.search"
    assert "rag.search" in captured["prompt"]


def test_plan_execution_force_inserts_rag_when_llm_picks_only_web() -> None:
    """LLM 直接给了 web.search 作为第 1 步时，planner 必须在最前面强制插入 rag.search。"""

    intent = IntentAnalysis(
        summary="今天的新闻",
        domain_hints=["web_realtime"],
        complexity="simple",
        need_tools=True,
    )

    def fake_llm(_prompt: str) -> str:
        return (
            '{"reasoning":"web only","steps":[{'
            '"title":"联网","intent":"查最新",'
            '"suggested_skill_id":"web.search",'
            '"suggested_arguments":{"query":"今天的新闻"},'
            '"expected_output":"新闻列表"}]}'
        )

    plan = plan_execution(
        intent=intent,
        registry=_FakeRegistry(_DUMMY_SPECS),
        llm_invoke=fake_llm,
    )
    assert plan.steps[0].suggested_skill_id == "rag.search"
    assert any(s.suggested_skill_id == "web.search" for s in plan.steps)


def test_heuristic_plan_always_starts_with_rag() -> None:
    intent = IntentAnalysis(
        summary="查文档",
        domain_hints=["knowledge_base"],
        need_tools=True,
    )
    plan = heuristic_plan(intent)
    assert len(plan.steps) >= 1
    assert plan.steps[0].suggested_skill_id == "rag.search"


def test_heuristic_plan_appends_web_when_realtime_hint() -> None:
    intent = IntentAnalysis(
        summary="最新", domain_hints=["web_realtime"], need_tools=True,
    )
    plan = heuristic_plan(intent)
    skill_ids = [s.suggested_skill_id for s in plan.steps]
    assert skill_ids[0] == "rag.search"
    assert "web.search" in skill_ids


def test_plan_step_to_dict_roundtrip() -> None:
    step = PlanStep(
        id="step-x",
        title="t",
        intent="i",
        suggested_skill_id="rag.search",
        suggested_arguments={"q": 1},
        expected_output="o",
    )
    d = step.to_dict()
    assert d["id"] == "step-x"
    assert d["status"] == "planned"
    assert d["suggested_arguments"] == {"q": 1}
