"""执行计划 Planner。

职责：
  - 读取 IntentAnalysis + 全量 skill catalog
  - 让 LLM 产出"先做什么再做什么"的步骤列表
  - 每个步骤包含：title、intent、suggested_skill_id、suggested_arguments、
    expected_output、status（planned）
  - 允许 "no_tool_step"：纯推理或等待 LLM 综合回答的步骤

设计要点：
  - 计划是"建议"，skill_router 执行时可以根据实时证据调整
  - 闲聊等 simple 场景允许产出空步骤列表（直接 finalize）
  - 输出严格 JSON，失败时有启发式兜底
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ops_rag_agent.agents.intent_analyzer import IntentAnalysis
from ops_rag_agent.agents.skill_router import build_skill_catalog_block
from ops_rag_agent.skills.registry import SkillRegistry


_PLANNER_PROMPT_TEMPLATE = """你是一个执行计划生成器（Planner）。
给你一份用户意图分析 + 可用 Skills 目录，请产出一份分步执行计划。

输出必须是严格 JSON，不要任何 Markdown 代码块包裹，不要多余解释。

OUTPUT_SCHEMA：
{{
  "reasoning": "用 1-2 句中文解释整体思路",
  "steps": [
    {{
      "title": "步骤简短标题（中文，<=20 字）",
      "intent": "这一步想达到什么，写 1 句中文",
      "suggested_skill_id": "<AVAILABLE_SKILLS 里的某个 id>",
      "suggested_arguments": {{...符合该 skill 的 argument_schema}},
      "expected_output": "预期能从这步拿到什么信息"
    }}
  ]
}}

强约束（本系统固定路由策略）：
  - 第 1 步必须是 rag.search（内部知识库检索），query 使用用户 summary 或最关键的子问题
  - 若 domain_hints 含 "web_realtime" 或用户明显需要联网最新信息（例如"最新""今天""实时"），
    可以追加第 2 步 web.search
  - 若 domain_hints 含 "ops_local" / "ops_remote"，可以在 rag.search 之后追加一个 ops.* skill
  - 不要在 plan 中硬编码 web.search 作为"rag 不够就联网"的兜底步骤；
    "rag 结果不足时再联网"的动态决策由 skill_router 在运行时根据观测做出
  - 最多 4 步；每步聚焦一个子问题；同一 skill 多步必须 arguments 不同
  - suggested_skill_id 必须严格等于下面 AVAILABLE_SKILLS 里的某个 id，不要自创
  - 不要把 final answer 写进 plan，只描述"要做什么"

INTENT_ANALYSIS:
{intent_json}

AVAILABLE_SKILLS（工具箱）:
{skill_catalog}

请输出严格 JSON：
"""


@dataclass
class PlanStep:
    id: str
    title: str
    intent: str
    suggested_skill_id: str = ""
    suggested_arguments: dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    status: str = "planned"  # planned / running / done / failed / skipped

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "intent": self.intent,
            "suggested_skill_id": self.suggested_skill_id,
            "suggested_arguments": dict(self.suggested_arguments),
            "expected_output": self.expected_output,
            "status": self.status,
        }


@dataclass
class ExecutionPlan:
    steps: list[PlanStep] = field(default_factory=list)
    reasoning: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "reasoning": self.reasoning,
            "created_at": self.created_at,
        }


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text


def _new_step_id() -> str:
    return "step-" + uuid.uuid4().hex[:8]


def parse_plan(raw_text: str) -> ExecutionPlan:
    text = _strip_fences(raw_text)
    match = _JSON_OBJECT_PATTERN.search(text)
    candidate = match.group(0) if match else text
    now = datetime.now(timezone.utc).isoformat()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return ExecutionPlan(
            steps=[],
            reasoning="llm_output_not_json",
            created_at=now,
        )
    if not isinstance(data, dict):
        return ExecutionPlan(steps=[], reasoning="llm_output_not_object", created_at=now)

    steps_raw = data.get("steps") or []
    if not isinstance(steps_raw, list):
        steps_raw = []

    steps: list[PlanStep] = []
    for item in steps_raw[:10]:
        if not isinstance(item, dict):
            continue
        args = item.get("suggested_arguments") or {}
        if not isinstance(args, dict):
            args = {}
        steps.append(
            PlanStep(
                id=_new_step_id(),
                title=str(item.get("title") or "").strip()[:80] or "未命名步骤",
                intent=str(item.get("intent") or "").strip()[:400],
                suggested_skill_id=str(item.get("suggested_skill_id") or "").strip(),
                suggested_arguments=args,
                expected_output=str(item.get("expected_output") or "").strip()[:400],
            )
        )

    return ExecutionPlan(
        steps=steps,
        reasoning=str(data.get("reasoning") or "").strip(),
        created_at=now,
    )


def plan_execution(
    *,
    intent: IntentAnalysis,
    registry: SkillRegistry,
    llm_invoke: Callable[[str], str],
    allowed_business_domains: Optional[list[str]] = None,
) -> ExecutionPlan:
    """基于意图分析和可用 skill 产出计划。

    新策略：所有请求强制至少跑一次 rag.search，是否再追加 web.search
    由 skill_router 根据 rag 结果动态决定（不在 plan 里硬编码兜底）。
    """

    catalog = build_skill_catalog_block(registry, allowed_business_domains=allowed_business_domains)
    prompt = _PLANNER_PROMPT_TEMPLATE.format(
        intent_json=json.dumps(intent.to_dict(), ensure_ascii=False, indent=2),
        skill_catalog=catalog,
    )
    raw = llm_invoke(prompt)
    plan = parse_plan(raw)

    # 兜底：若 LLM 漏写了 rag.search 作为第 1 步，强制注入
    plan = _ensure_rag_first(plan, intent)
    return plan


def _ensure_rag_first(plan: ExecutionPlan, intent: IntentAnalysis) -> ExecutionPlan:
    """若 plan 第 1 步不是 rag.search，则在最前面插入一个 rag.search 步骤。"""

    has_rag_first = (
        plan.steps
        and plan.steps[0].suggested_skill_id == "rag.search"
    )
    if has_rag_first:
        return plan

    rag_step = PlanStep(
        id=_new_step_id(),
        title="内部知识库检索",
        intent="先在内部 RAG 知识库中查找与用户问题相关的资料",
        suggested_skill_id="rag.search",
        suggested_arguments={
            "query": intent.summary or (intent.sub_questions[0] if intent.sub_questions else ""),
            "top_k": 5,
        },
        expected_output="若干条带分数的内部文档片段",
    )
    new_steps = [rag_step, *plan.steps]
    return ExecutionPlan(
        steps=new_steps[:10],
        reasoning=plan.reasoning or "强制先走内部 RAG 检索",
        created_at=plan.created_at,
    )


def heuristic_plan(intent: IntentAnalysis) -> ExecutionPlan:
    """LLM 不可用时的兜底：永远先 rag.search，再根据 hints 决定是否追加。"""

    now = datetime.now(timezone.utc).isoformat()
    steps: list[PlanStep] = []

    # 第 1 步必跑 rag.search
    steps.append(
        PlanStep(
            id=_new_step_id(),
            title="内部知识库检索",
            intent="先在内部 RAG 知识库中查找相关资料",
            suggested_skill_id="rag.search",
            suggested_arguments={
                "query": intent.summary or (intent.sub_questions[0] if intent.sub_questions else ""),
                "top_k": 5,
            },
            expected_output="若干条带分数的内部文档片段",
        )
    )

    # 显式 web_realtime hint 时追加联网搜索（其余情况由 router 动态决定是否要追加）
    if any(h == "web_realtime" for h in intent.domain_hints):
        steps.append(
            PlanStep(
                id=_new_step_id(),
                title="联网搜索补充",
                intent="联网搜索最新/实时信息以补充内部 RAG",
                suggested_skill_id="web.search",
                suggested_arguments={"query": intent.summary or ""},
                expected_output="若干条带 URL 的公网搜索结果",
            )
        )

    # ops_local / ops_remote hint 时追加一次 ops 快照
    hint_to_ops = {
        "ops_local": "ops.local.snapshot",
        "ops_remote": "ops.remote.snapshot",
    }
    for hint, skill_id in hint_to_ops.items():
        if hint in intent.domain_hints:
            steps.append(
                PlanStep(
                    id=_new_step_id(),
                    title=f"{hint} 快速检查",
                    intent=f"使用 {skill_id} 处理 {hint} 相关子问题",
                    suggested_skill_id=skill_id,
                    suggested_arguments={},
                    expected_output="本机/远端运行态快照",
                )
            )

    return ExecutionPlan(
        steps=steps,
        reasoning="heuristic_plan (LLM 不可用时的兜底)",
        created_at=now,
    )
