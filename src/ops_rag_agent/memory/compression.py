from __future__ import annotations

import json
from typing import Any


def split_for_compression(messages: list[dict[str, Any]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Protect head + tail, compress only the middle zone.

    Head protection:
    - first system prompt
    - first human message
    - first assistant message
    - first tool interaction
    Tail protection:
    - last 4 rounds (simplified here as last 8 messages)
    """

    if len(messages) <= 12:
        return messages, [], []

    head = messages[:4]
    tail = messages[-8:]
    middle = messages[4:-8]
    return head, middle, tail


def should_compress(token_usage: dict[str, int], max_context_tokens: int, trigger_ratio: float) -> bool:
    used = token_usage.get("total", 0)
    return used >= int(max_context_tokens * trigger_ratio)


_LLM_SUMMARY_SYSTEM = (
    "你是一个对话历史压缩器。用户会给你一段中间历史消息 JSON，"
    "请用中文输出一份结构化的滚动摘要，保留：\n"
    "  task_goal: 当前对话的总体目标\n"
    "  confirmed_facts: 已经被确认/采集到的关键事实（列表）\n"
    "  actions_taken: 已经执行过的关键动作/命令/工具调用（列表）\n"
    "  pending_questions: 尚未回答或未解决的问题（列表）\n"
    "  constraints: 对后续对话的约束（例如权限、审批、只读等）\n"
    "  next_best_actions: 下一步最合理的动作建议（列表）\n"
    "每个列表最多 8 条，语言要精炼，不要复述原文。"
    "严格使用上述 6 个 key 作为段落前缀。"
)


def _llm_summarize_middle(
    middle: list[dict[str, Any]],
    *,
    detail_level: int = 1,
    reflection_feedback: list[str] | None = None,
) -> str | None:
    """用 LLM 生成摘要；失败返回 None，调用方回退到模板版本。"""

    try:
        # 延迟 import，避免测试/导入期依赖 LangChain
        from ops_rag_agent.models.factory import build_chat_llm
    except Exception:
        return None

    try:
        payload = {
            "detail_level": detail_level,
            "reflection_feedback": list(reflection_feedback or []),
            "messages": [
                {
                    "role": str(item.get("role", "unknown")),
                    "content": str(item.get("content", ""))[:2000],
                }
                for item in middle
            ],
        }
        prompt = (
            _LLM_SUMMARY_SYSTEM
            + "\n\nMIDDLE_MESSAGES_JSON:\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        llm = build_chat_llm()
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", "") if resp is not None else ""
        if isinstance(text, list):
            text = "\n".join(
                str(part.get("text", part)) if isinstance(part, dict) else str(part)
                for part in text
            )
        text = str(text or "").strip()
        return text or None
    except Exception:
        return None


def summarize_middle_zone(
    middle: list[dict[str, Any]],
    *,
    detail_level: int = 1,
    reflection_feedback: list[str] | None = None,
) -> str:
    if not middle:
        return ""

    # 优先 LLM 生成
    llm_summary = _llm_summarize_middle(
        middle,
        detail_level=detail_level,
        reflection_feedback=reflection_feedback,
    )
    if llm_summary:
        return llm_summary

    # ---- 回退：模板化摘要，保证链路可用 ----
    actions: list[str] = []
    facts: list[str] = []
    pending: list[str] = []
    constraints: list[str] = ["preserve head and tail zones"]
    feedback = reflection_feedback or []

    for item in middle:
        role = item.get("role", "unknown")
        content = str(item.get("content", ""))[:200]
        if role in {"tool", "function"}:
            actions.append(content)
        else:
            facts.append(content)
            if "?" in content:
                pending.append(content)

    fact_limit = 5 + max(detail_level - 1, 0) * 2
    action_limit = 5 + max(detail_level - 1, 0) * 2
    if feedback:
        constraints.append("reflection_feedback=" + "; ".join(feedback[:3]))

    return (
        "task_goal: continue current workflow\n"
        f"confirmed_facts: {facts[:fact_limit]}\n"
        f"actions_taken: {actions[:action_limit]}\n"
        f"pending_questions: {pending[:3]}\n"
        f"constraints: {constraints}\n"
        "next_best_actions: continue from latest protected tail"
    )


def reflect_compression_summary(
    summary: str,
    middle: list[dict[str, Any]],
) -> dict[str, Any]:
    if not summary:
        return {
            "status": "skipped",
            "missing_sections": ["task_goal", "confirmed_facts", "actions_taken"],
            "feedback": ["summary is empty"],
            "passed": False,
        }

    required_sections = [
        "task_goal:",
        "confirmed_facts:",
        "actions_taken:",
        "pending_questions:",
        "constraints:",
        "next_best_actions:",
    ]
    missing_sections = [section[:-1] for section in required_sections if section not in summary]

    tool_messages = [
        str(item.get("content", ""))[:80]
        for item in middle
        if item.get("role") in {"tool", "function"} and item.get("content")
    ]
    missing_tool_results = bool(tool_messages) and "actions_taken: []" in summary

    feedback: list[str] = []
    if missing_sections:
        feedback.append("missing required sections: " + ", ".join(missing_sections))
    if tool_messages and not any(snippet in summary for snippet in tool_messages[:2]):
        feedback.append("summary should retain important tool results")
    if missing_tool_results:
        feedback.append("actions_taken should not be empty when tool outputs exist")

    return {
        "status": "passed" if not feedback else "failed",
        "missing_sections": missing_sections,
        "feedback": feedback,
        "passed": not feedback,
    }


def recompress_with_reflection(
    middle: list[dict[str, Any]],
    reflection_feedback: list[str],
    retry_count: int,
) -> str:
    summary = summarize_middle_zone(
        middle,
        detail_level=retry_count + 2,
        reflection_feedback=reflection_feedback,
    )
    if not reflection_feedback:
        return summary
    if "reflection_feedback=" in summary:
        final_summary = summary
    else:
        feedback_line = "reflection_feedback=" + "; ".join(reflection_feedback[:3])
        if "constraints:" in summary:
            lines = summary.splitlines()
            for index, line in enumerate(lines):
                if line.startswith("constraints:"):
                    lines.insert(index + 1, feedback_line)
                    final_summary = "\n".join(lines)
                    break
            else:
                final_summary = summary.rstrip() + "\n" + feedback_line
        else:
            final_summary = summary.rstrip() + "\n" + feedback_line

    tool_snippets = [
        str(item.get("content", "")).strip()[:120]
        for item in middle
        if item.get("role") in {"tool", "function"} and str(item.get("content", "")).strip()
    ]
    for snippet in tool_snippets[:2]:
        if snippet and snippet not in final_summary:
            final_summary = final_summary.rstrip() + "\n" + f"source_evidence={snippet}"
    return final_summary
