"""skill_router 单测：ReAct 循环 / finalize / 死循环 / 审批挂起 / 错误解析。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from ops_rag_agent.agents.skill_router import (
    ApprovalRequest,
    ScratchpadEntry,
    parse_router_action,
    parse_router_decision,
    run_skill_router,
)
from ops_rag_agent.skills.base import SkillKind, SkillSpec
from ops_rag_agent.skills.registry import SkillRegistry


# ---------- fakes ----------


class _ArgsModel(BaseModel):
    query: str
    top_k: int = 5


class _ResultModel(BaseModel):
    ok: bool
    message: str


class _CmdArgsModel(BaseModel):
    cmd: str


@dataclass
class _FakeSkill:
    spec: SkillSpec
    output: str = "ok"
    raise_exc: bool = False

    def invoke(self, arguments: dict[str, Any]) -> str:
        if self.raise_exc:
            raise RuntimeError("boom")
        return f"{self.output}|{json.dumps(arguments, sort_keys=True)}"


def _mk_registry(*skills: _FakeSkill) -> SkillRegistry:
    reg = SkillRegistry()
    for s in skills:
        reg.register(s)
    return reg


def _low_risk_skill(
    skill_id: str,
    output: str = "ok",
    *,
    business_domain: str = "ops",
) -> _FakeSkill:
    return _FakeSkill(
        spec=SkillSpec(
            skill_id=skill_id,
            name=skill_id,
            description="dummy low risk",
            business_domain=business_domain,
            kind=SkillKind.REGULAR,
            requires_approval=False,
            is_readonly=True,
            risk_level="low",
            when_to_use="test",
        ),
        output=output,
    )


def _high_risk_skill(skill_id: str) -> _FakeSkill:
    return _FakeSkill(
        spec=SkillSpec(
            skill_id=skill_id,
            name=skill_id,
            description="dummy high risk",
            business_domain="ops",
            kind=SkillKind.REGULAR,
            requires_approval=True,
            is_readonly=True,
            risk_level="high",
            when_to_use="test",
        )
    )


def _typed_skill(skill_id: str = "ops.typed") -> _FakeSkill:
    return _FakeSkill(
        spec=SkillSpec(
            skill_id=skill_id,
            name=skill_id,
            description="typed skill",
            business_domain="ops",
            kind=SkillKind.REGULAR,
            requires_approval=False,
            is_readonly=True,
            risk_level="low",
            when_to_use="test typed validation",
            argument_model=_ArgsModel,
        ),
        output="TYPED_OUT",
    )


@dataclass
class _FakeRagSkill:
    spec: SkillSpec = SkillSpec(
        skill_id="rag.search",
        name="rag.search",
        description="dummy rag",
        business_domain="knowledge",
        kind=SkillKind.REGULAR,
        requires_approval=False,
        is_readonly=True,
        risk_level="low",
        when_to_use="test rag first",
        argument_model=_ArgsModel,
    )

    def invoke(self, arguments: dict[str, Any]) -> str:
        return json.dumps(
            {
                "query": arguments["query"],
                "top_k": arguments.get("top_k", 5),
                "results": [{"doc_id": "d1", "text": "kb"}],
            },
            ensure_ascii=False,
        )


def _high_risk_typed_skill(skill_id: str = "ops.secure") -> _FakeSkill:
    return _FakeSkill(
        spec=SkillSpec(
            skill_id=skill_id,
            name=skill_id,
            description="dummy high risk typed",
            business_domain="ops",
            kind=SkillKind.REGULAR,
            requires_approval=True,
            is_readonly=False,
            risk_level="high",
            when_to_use="test approval resume validation",
            argument_model=_CmdArgsModel,
        )
    )


def _scripted_llm(responses: list[str]):
    it = iter(responses)

    def _invoke(prompt: str) -> str:  # noqa: ARG001
        try:
            return next(it)
        except StopIteration:
            return json.dumps({"action": "finalize", "thought": "oom", "final_answer": "done"})

    return _invoke


# ---------- parse_router_decision ----------


def test_parse_valid_call_skill() -> None:
    data = parse_router_decision(
        '{"action": "call_skill", "skill_id": "x", "arguments": {}}'
    )
    assert data["action"] == "call_skill"


def test_parse_finalize_with_markdown_fence() -> None:
    raw = "```json\n{\"action\": \"finalize\", \"final_answer\": \"ok\"}\n```"
    data = parse_router_decision(raw)
    assert data["action"] == "finalize"
    assert data["final_answer"] == "ok"


def test_parse_empty_returns_error() -> None:
    data = parse_router_decision("")
    assert data["action"] == "error"


def test_parse_bad_json_returns_error() -> None:
    data = parse_router_decision("not a json")
    assert data["action"] == "error"


def test_parse_native_tool_call_payload_to_router_action() -> None:
    action = parse_router_action(
        {
            "name": "call_skill",
            "arguments": {
                "skill_id": "ops.a",
                "arguments": {"k": 1},
                "plan_step_id": "step-1",
            },
        }
    )

    assert action.action == "call_skill"
    assert action.skill_id == "ops.a"
    assert action.arguments == {"k": 1}
    assert action.plan_step_id == "step-1"
    assert action.source == "native_tool_call"


def test_parse_ai_message_tool_call_to_router_action() -> None:
    message = AIMessage(
        content="优先调用 skill",
        tool_calls=[
            {
                "name": "call_skill",
                "args": {
                    "skill_id": "ops.a",
                    "arguments": {"k": 1},
                    "plan_step_id": "step-ai",
                },
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )

    action = parse_router_action(message)

    assert action.action == "call_skill"
    assert action.skill_id == "ops.a"
    assert action.arguments == {"k": 1}
    assert action.plan_step_id == "step-ai"
    assert action.source == "native_tool_call"


def test_skill_spec_manifest_exposes_runtime_contract() -> None:
    spec = SkillSpec(
        skill_id="web.demo",
        name="web-demo",
        description="demo",
        business_domain="general",
        kind=SkillKind.REGULAR,
        argument_model=_ArgsModel,
        result_model=_ResultModel,
    )

    manifest = spec.to_manifest()

    assert manifest["argument_model"].endswith("._ArgsModel")
    assert manifest["result_model"].endswith("._ResultModel")
    assert manifest["supports_runtime_validation"] is True
    assert manifest["supports_structured_output"] is True
    assert manifest["argument_schema"]["properties"]["query"]["type"] == "string"
    assert manifest["result_schema"]["properties"]["ok"]["type"] == "boolean"


# ---------- 循环主流程 ----------


def test_finalize_immediately() -> None:
    reg = _mk_registry(_low_risk_skill("ops.a"))
    llm = _scripted_llm(
        [
            json.dumps(
                {"action": "finalize", "thought": "no need", "final_answer": "hi"}
            )
        ]
    )
    result = run_skill_router(
        user_query="你好", registry=reg, llm_invoke=llm, max_iterations=5
    )
    assert result.final_answer == "hi"
    assert result.stop_reason == "finalize"
    assert result.iterations == 1


def test_call_low_risk_then_finalize() -> None:
    reg = _mk_registry(_low_risk_skill("ops.a", output="A_OUT"))
    llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_skill",
                    "thought": "try a",
                    "skill_id": "ops.a",
                    "arguments": {"k": 1},
                }
            ),
            json.dumps(
                {
                    "action": "finalize",
                    "thought": "done",
                    "final_answer": "got A_OUT",
                }
            ),
        ]
    )
    result = run_skill_router(
        user_query="run a", registry=reg, llm_invoke=llm, max_iterations=5
    )
    assert result.stop_reason == "finalize"
    assert result.iterations == 2
    assert any(e.skill_id == "ops.a" for e in result.scratchpad)
    assert "A_OUT" in result.scratchpad[0].observation
    assert result.scratchpad[0].validation["status"] == "skipped"
    assert result.scratchpad[0].observation_data["status"] == "success"
    assert any(
        item["phase"] == "execute" for item in result.scratchpad[0].runtime_audit
    )


def test_invalid_arguments_returns_structured_error_and_allows_recovery() -> None:
    reg = _mk_registry(_typed_skill())
    llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_skill",
                    "thought": "参数先试错",
                    "skill_id": "ops.typed",
                    "arguments": {},
                }
            ),
            json.dumps(
                {
                    "action": "finalize",
                    "thought": "拿到错误后结束",
                    "final_answer": "参数可修正",
                }
            ),
        ]
    )
    result = run_skill_router(
        user_query="run typed", registry=reg, llm_invoke=llm, max_iterations=5
    )

    assert result.stop_reason == "finalize"
    first = result.scratchpad[0]
    assert first.skill_id == "ops.typed"
    assert first.status == "failed"
    assert first.validation["status"] == "failed"
    assert first.observation_data["error"]["code"] == "invalid_arguments"
    assert first.observation_data["structured_output"]["issues"][0]["path"] == ["query"]


def test_rag_first_policy_blocks_and_model_can_recover() -> None:
    reg = _mk_registry(_FakeRagSkill(), _low_risk_skill("ops.a"))
    llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_skill",
                    "thought": "先跑 ops",
                    "skill_id": "ops.a",
                    "arguments": {"k": 1},
                }
            ),
            json.dumps(
                {
                    "action": "call_skill",
                    "thought": "先查知识库",
                    "skill_id": "rag.search",
                    "arguments": {"query": "kb", "top_k": 3},
                }
            ),
            json.dumps(
                {
                    "action": "finalize",
                    "thought": "已有证据",
                    "final_answer": "已按 rag-first 恢复",
                }
            ),
        ]
    )
    result = run_skill_router(
        user_query="need ops", registry=reg, llm_invoke=llm, max_iterations=5
    )

    assert result.stop_reason == "finalize"
    first = result.scratchpad[0]
    second = result.scratchpad[1]
    assert first.status == "blocked"
    assert first.observation_data["error"]["code"] == "rag_search_required"
    assert second.skill_id == "rag.search"
    assert second.status == "done"
    assert second.observation_data["status"] == "success"


def test_router_prompt_uses_shortlist_instead_of_full_catalog() -> None:
    reg = _mk_registry(
        _low_risk_skill("rag.search", business_domain="knowledge"),
        _low_risk_skill("ops.local.snapshot", business_domain="ops"),
        _low_risk_skill("web.search", business_domain="general"),
        _low_risk_skill("misc.general", business_domain="general"),
    )
    prompts: list[str] = []

    def _capture_prompt(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps(
            {
                "action": "finalize",
                "thought": "enough",
                "final_answer": "done",
            }
        )

    run_skill_router(
        user_query="帮我看下本机端口",
        registry=reg,
        llm_invoke=_capture_prompt,
        max_iterations=3,
        intent_analysis={"domain_hints": ["knowledge_base", "ops_local"]},
        execution_plan={
            "steps": [
                {
                    "id": "step-1",
                    "suggested_skill_id": "ops.local.snapshot",
                    "suggested_arguments": {},
                }
            ]
        },
    )

    prompt = prompts[0]
    assert "skill_id: rag.search" in prompt
    assert "skill_id: ops.local.snapshot" in prompt
    assert "skill_id: web.search" not in prompt
    assert "skill_id: misc.general" not in prompt


def test_router_accepts_native_tool_call_payloads() -> None:
    reg = _mk_registry(_low_risk_skill("ops.a", output="A_OUT"))
    llm = _scripted_llm(
        [
            {
                "name": "call_skill",
                "arguments": {
                    "skill_id": "ops.a",
                    "arguments": {"k": 1},
                    "plan_step_id": "step-native",
                },
            },
            {
                "name": "finalize",
                "arguments": {
                    "final_answer": "native ok",
                    "stop_reason": "enough_evidence",
                },
            },
        ]
    )

    result = run_skill_router(
        user_query="run native",
        registry=reg,
        llm_invoke=llm,
        max_iterations=5,
    )

    assert result.stop_reason == "enough_evidence"
    assert result.final_answer == "native ok"
    assert result.scratchpad[0].plan_step_id == "step-native"
    assert "A_OUT" in result.scratchpad[0].observation


def test_high_risk_triggers_pending_approval() -> None:
    reg = _mk_registry(_high_risk_skill("ops.danger"))
    llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_skill",
                    "thought": "need danger",
                    "skill_id": "ops.danger",
                    "arguments": {"cmd": "rm"},
                }
            )
        ]
    )
    result = run_skill_router(
        user_query="do danger", registry=reg, llm_invoke=llm, max_iterations=5
    )
    assert result.stop_reason == "pending_approval"
    assert isinstance(result.pending_approval, ApprovalRequest)
    assert result.pending_approval.skill_id == "ops.danger"
    assert result.pending_approval.risk_level == "high"
    assert result.scratchpad[0].observation_data["status"] == "pending_approval"
    assert any(
        item["phase"] == "policy" for item in result.scratchpad[0].runtime_audit
    )


def test_resume_from_approval_approved() -> None:
    reg = _mk_registry(_high_risk_skill("ops.danger"))
    # 恢复时直接跑 ops.danger，下一步就 finalize
    llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "finalize",
                    "thought": "collected enough",
                    "final_answer": "ok",
                }
            )
        ]
    )
    result = run_skill_router(
        user_query="do danger",
        registry=reg,
        llm_invoke=llm,
        max_iterations=5,
        approval_resume={
            "approved": True,
            "skill_id": "ops.danger",
            "arguments": {"cmd": "rm"},
        },
    )
    assert result.stop_reason == "finalize"
    # 第一条应该是审批恢复后立即执行的记录
    assert result.scratchpad[0].skill_id == "ops.danger"
    assert result.scratchpad[0].action == "call_skill"


def test_resume_from_approval_approved_still_uses_runtime_validation() -> None:
    reg = _mk_registry(_high_risk_typed_skill())
    llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "finalize",
                    "thought": "看到结构化错误后结束",
                    "final_answer": "resume checked",
                }
            )
        ]
    )
    result = run_skill_router(
        user_query="resume secure",
        registry=reg,
        llm_invoke=llm,
        max_iterations=5,
        approval_resume={
            "approved": True,
            "skill_id": "ops.secure",
            "arguments": {},
        },
    )

    first = result.scratchpad[0]
    assert result.stop_reason == "finalize"
    assert first.skill_id == "ops.secure"
    assert first.validation["status"] == "failed"
    assert first.observation_data["error"]["code"] == "invalid_arguments"


def test_resume_from_approval_rejected_logged() -> None:
    reg = _mk_registry(_high_risk_skill("ops.danger"), _low_risk_skill("ops.a"))
    llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "finalize",
                    "thought": "switch plan",
                    "final_answer": "switched",
                }
            )
        ]
    )
    result = run_skill_router(
        user_query="q",
        registry=reg,
        llm_invoke=llm,
        max_iterations=5,
        approval_resume={
            "approved": False,
            "skill_id": "ops.danger",
            "arguments": {"cmd": "rm"},
        },
    )
    assert result.stop_reason == "finalize"
    assert result.scratchpad[0].action == "approval_rejected"
    assert result.scratchpad[0].observation_data["status"] == "blocked"
    assert result.scratchpad[0].observation_data["error"]["code"] == "approval_rejected"


def test_loop_detected_on_same_skill_args() -> None:
    reg = _mk_registry(_low_risk_skill("ops.a"))
    # LLM 连续 3 次要同一个命令，期望第 3 次进入 loop_detected
    call = json.dumps(
        {
            "action": "call_skill",
            "thought": "again",
            "skill_id": "ops.a",
            "arguments": {"k": 1},
        }
    )
    llm = _scripted_llm([call, call, call, call])
    result = run_skill_router(
        user_query="loop", registry=reg, llm_invoke=llm, max_iterations=5
    )
    assert result.stop_reason == "loop_detected"
    assert result.scratchpad[-1].observation_data["error"]["code"] == "loop_detected"


def test_max_iterations_bound() -> None:
    reg = _mk_registry(_low_risk_skill("ops.a"), _low_risk_skill("ops.b"))
    # LLM 交替调 a/b 永不 finalize
    alt = [
        json.dumps(
            {
                "action": "call_skill",
                "thought": "x",
                "skill_id": "ops.a" if i % 2 == 0 else "ops.b",
                "arguments": {"i": i},
            }
        )
        for i in range(10)
    ]
    llm = _scripted_llm(alt)
    result = run_skill_router(
        user_query="q", registry=reg, llm_invoke=llm, max_iterations=4
    )
    assert result.stop_reason == "max_iterations"
    assert result.iterations == 4


def test_unknown_skill_recorded_and_continues() -> None:
    reg = _mk_registry(_low_risk_skill("ops.a"))
    llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_skill",
                    "thought": "try ghost",
                    "skill_id": "ops.ghost",
                    "arguments": {},
                }
            ),
            json.dumps(
                {
                    "action": "finalize",
                    "thought": "stop",
                    "final_answer": "no way",
                }
            ),
        ]
    )
    result = run_skill_router(
        user_query="q", registry=reg, llm_invoke=llm, max_iterations=5
    )
    assert result.stop_reason == "finalize"
    assert any(
        e.status == "failed" and "unknown_skill_id" in e.observation
        for e in result.scratchpad
    )
    failed = next(e for e in result.scratchpad if e.status == "failed")
    assert failed.observation_data["status"] == "failed"
    assert failed.observation_data["error"]["code"] == "unknown_skill_id"
