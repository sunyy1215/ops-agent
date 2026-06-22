"""意图分析 Agent。

职责：
  - 接收用户原始 query
  - 用 LLM 做结构化意图分析：问题摘要、子问题拆解、领域提示、复杂度评估、是否需要工具
  - 输出 dict 结构供 planner / skill_router / 前端使用

不接 LangGraph，纯函数 + dataclass，方便单测。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


_INTENT_PROMPT_TEMPLATE = """你是一个意图分析助手。
用户给出一个自然语言请求，你的任务是：在不调用任何工具的情况下，
仅基于语义快速做"意图分析"，输出严格 JSON。

工作要求：
  1. 不要回答用户的问题本身，只做"分析"
  2. 输出必须是合法 JSON，不要 Markdown 代码块包裹
  3. 输出字段必须严格匹配下面的 Schema

OUTPUT_SCHEMA（严格遵守）：
{{
  "summary": "用一句中文说明用户究竟想知道什么",
  "sub_questions": [
    "把用户问题拆成 1~5 个互不重叠的子问题，每条 1 句中文"
  ],
  "domain_hints": [
    "可选标签，描述这次请求涉及的领域，例如：knowledge_base / web_realtime / ops_local / ops_remote / coding / general_qa"
  ],
  "complexity": "simple" | "moderate" | "complex",
  "need_tools": true,
  "reasoning": "用 1-2 句中文解释你为什么这么判断（写给开发者看，简短）"
}}

判断标准：
  - need_tools 必须始终为 true：本系统强制先查内部知识库（rag.search），
    必要时再追加联网搜索（web.search）
  - simple：单一明确意图，可能 1 步内拿到结果（例如概念/原理/术语解释）
  - moderate：典型情况，需要 1-3 步工具检索
  - complex：含多个子问题、跨领域、需要联动多个工具
  - 若问题明显涉及"最新 / 今天 / 实时 / 公网资讯"，在 domain_hints 加 "web_realtime"
  - 若问题涉及"本机/这台机器/本地端口/本地进程"，在 domain_hints 加 "ops_local"
  - 否则默认在 domain_hints 加 "knowledge_base"，让系统先去内部 RAG 检索

USER_QUERY:
{user_query}

请输出严格 JSON：
"""


@dataclass
class IntentAnalysis:
    summary: str = ""
    sub_questions: list[str] = field(default_factory=list)
    domain_hints: list[str] = field(default_factory=list)
    complexity: str = "moderate"  # simple / moderate / complex
    # 在新流水线中，need_tools 强制为 True：所有请求都先尝试内部 RAG，
    # 必要时再追加联网搜索。保留字段做兼容/前端展示用。
    need_tools: bool = True
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "sub_questions": list(self.sub_questions),
            "domain_hints": list(self.domain_hints),
            "complexity": self.complexity,
            "need_tools": self.need_tools,
            "reasoning": self.reasoning,
        }


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text


def parse_intent(raw_text: str) -> IntentAnalysis:
    """从 LLM 输出抽取 IntentAnalysis；失败时返回降级版。"""

    text = _strip_fences(raw_text)
    match = _JSON_OBJECT_PATTERN.search(text)
    candidate = match.group(0) if match else text
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return IntentAnalysis(
            summary=(raw_text or "").strip()[:200] or "(无法解析意图)",
            sub_questions=[],
            domain_hints=["knowledge_base"],
            complexity="moderate",
            need_tools=True,
            reasoning="llm_output_not_json",
        )
    if not isinstance(data, dict):
        return IntentAnalysis(
            reasoning="llm_output_not_object",
            domain_hints=["knowledge_base"],
            need_tools=True,
        )

    sub = data.get("sub_questions") or []
    if not isinstance(sub, list):
        sub = []
    sub = [str(item).strip() for item in sub if str(item).strip()]

    hints = data.get("domain_hints") or []
    if not isinstance(hints, list):
        hints = []
    hints = [str(item).strip() for item in hints if str(item).strip()]
    # 不允许出现 chitchat：本系统不再有"纯闲聊短路"路径
    hints = [h for h in hints if h.lower() != "chitchat"]
    if not hints:
        hints = ["knowledge_base"]

    complexity = str(data.get("complexity") or "moderate").strip().lower()
    if complexity not in {"simple", "moderate", "complex"}:
        complexity = "moderate"

    # need_tools 强制为 True：所有请求都先走 rag.search
    return IntentAnalysis(
        summary=str(data.get("summary") or "").strip(),
        sub_questions=sub[:8],
        domain_hints=hints[:8],
        complexity=complexity,
        need_tools=True,
        reasoning=str(data.get("reasoning") or "").strip(),
    )


def analyze_intent(
    *,
    user_query: str,
    llm_invoke: Callable[[str], str],
) -> IntentAnalysis:
    """对用户请求做一次意图分析。"""

    query = (user_query or "").strip()
    if not query:
        return IntentAnalysis(
            summary="(空请求)",
            sub_questions=[],
            domain_hints=["knowledge_base"],
            complexity="simple",
            need_tools=True,
            reasoning="empty_user_query",
        )

    prompt = _INTENT_PROMPT_TEMPLATE.format(user_query=query)
    raw = llm_invoke(prompt)
    return parse_intent(raw)


# ---------- 简易启发式兜底（LLM 不可用时） ----------


def heuristic_intent(user_query: str) -> IntentAnalysis:
    """LLM 失败时的本地兜底分析，仅基于关键词粗判。

    注意：本系统不再保留"纯闲聊"短路路径，所有请求都至少先尝试一次内部 RAG。
    """

    q = (user_query or "").strip()
    if not q:
        return IntentAnalysis(
            summary="(空请求)",
            domain_hints=["knowledge_base"],
            complexity="simple",
            need_tools=True,
            reasoning="empty",
        )
    lower = q.lower()
    hints: list[str] = []
    if any(k in lower for k in ("最新", "今天", "新闻", "官网", "changelog", "release")):
        hints.append("web_realtime")
    if any(k in q for k in ("怎么排查", "故障", "运行", "运维", "进程", "端口", "本机", "这台")):
        hints.append("ops_local")
    if any(k in q for k in ("文档", "手册", "playbook", "知识库", "FAQ")):
        hints.append("knowledge_base")
    # 默认始终带上 knowledge_base，让 rag.search 必跑一次
    if "knowledge_base" not in hints:
        hints.insert(0, "knowledge_base")
    return IntentAnalysis(
        summary=q[:200],
        sub_questions=[q],
        domain_hints=hints,
        complexity="moderate",
        need_tools=True,
        reasoning="heuristic_fallback",
    )
