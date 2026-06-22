"""单测：intent_analyzer（意图分析 agent）。"""

from __future__ import annotations

from ops_rag_agent.agents.intent_analyzer import (
    IntentAnalysis,
    analyze_intent,
    heuristic_intent,
    parse_intent,
)


def test_parse_intent_accepts_clean_json() -> None:
    raw = """{
      "summary": "用户想查 k8s pod 重启原因",
      "sub_questions": ["pod 为什么重启", "如何查 lastState"],
      "domain_hints": ["ops_local", "knowledge_base"],
      "complexity": "moderate",
      "need_tools": true,
      "reasoning": "需要访问本机或查 playbook"
    }"""
    intent = parse_intent(raw)
    assert intent.summary.startswith("用户想查")
    assert len(intent.sub_questions) == 2
    assert intent.complexity == "moderate"
    assert intent.need_tools is True


def test_parse_intent_tolerates_markdown_fence_and_strips_chitchat() -> None:
    """LLM 输出 chitchat 时，本系统不再保留该 hint，且 need_tools 强制为 True。"""

    raw = """```json
{"summary": "hi", "sub_questions": [], "domain_hints": ["chitchat"],
 "complexity": "simple", "need_tools": false, "reasoning": "greeting"}
```"""
    intent = parse_intent(raw)
    assert intent.complexity == "simple"
    # 系统不再保留 chitchat 短路：need_tools 强制 True
    assert intent.need_tools is True
    # chitchat hint 被剥离，回退到 knowledge_base
    assert "chitchat" not in intent.domain_hints
    assert "knowledge_base" in intent.domain_hints


def test_parse_intent_rejects_garbage_but_does_not_crash() -> None:
    intent = parse_intent("not a json at all")
    assert isinstance(intent, IntentAnalysis)
    assert intent.reasoning == "llm_output_not_json"
    assert intent.need_tools is True
    assert "knowledge_base" in intent.domain_hints


def test_analyze_intent_empty_query_returns_knowledge_base() -> None:
    intent = analyze_intent(user_query="  ", llm_invoke=lambda _p: "{}")
    # 空 query 也仍然先走 knowledge_base，由 router 出兜底回复
    assert intent.need_tools is True
    assert "knowledge_base" in intent.domain_hints
    assert "chitchat" not in intent.domain_hints


def test_analyze_intent_invokes_llm_with_prompt() -> None:
    captured: dict[str, str] = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return (
            '{"summary":"查 k8s pod 重启","sub_questions":["为什么重启"],'
            '"domain_hints":["ops_local"],"complexity":"moderate",'
            '"need_tools":true,"reasoning":"need tools"}'
        )

    intent = analyze_intent(user_query="我的 pod 一直重启怎么办", llm_invoke=fake_llm)
    assert intent.need_tools is True
    assert intent.complexity == "moderate"
    assert "我的 pod 一直重启怎么办" in captured["prompt"]


def test_heuristic_intent_short_query_falls_back_to_knowledge_base() -> None:
    """短 query 不再走 chitchat 短路，仍然先去 RAG 查。"""

    intent = heuristic_intent("你好啊")
    assert intent.need_tools is True
    assert "chitchat" not in intent.domain_hints
    assert "knowledge_base" in intent.domain_hints


def test_heuristic_intent_classifies_ops_keywords() -> None:
    intent = heuristic_intent("本机端口被占用，帮我查一下进程故障")
    assert intent.need_tools is True
    assert "ops_local" in intent.domain_hints
    # knowledge_base 仍会被自动加上，确保至少先走一次 rag
    assert "knowledge_base" in intent.domain_hints
