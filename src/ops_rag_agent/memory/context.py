"""共享的 prompt 上下文构造：把 rolling_summary、long_term_memory_hits、最近若干轮
对话历史拼成一段 CONTEXT_BLOCK，供各 agent 插入到 prompt 前面，实现"历史摘要+最近
原文+本轮输入"的三段式结构。

这是压缩方案真正落地的关键一环：memory_compressor 把中段历史压缩为 rolling_summary，
此处负责把 rolling_summary 回灌进下一轮 LLM prompt。
"""

from __future__ import annotations

import json
from typing import Any, Iterable


def _msg_to_dict(msg: Any) -> dict[str, Any]:
    """兼容 LangChain 消息对象与 dict 两种形态。"""

    if isinstance(msg, dict):
        role = str(msg.get("role") or msg.get("type") or "user")
        content = msg.get("content", "")
    else:
        # LangChain BaseMessage: AIMessage / HumanMessage / SystemMessage / ToolMessage
        cls = msg.__class__.__name__
        role_map = {
            "SystemMessage": "system",
            "HumanMessage": "user",
            "AIMessage": "assistant",
            "ToolMessage": "tool",
            "FunctionMessage": "function",
        }
        role = role_map.get(cls, cls.lower())
        content = getattr(msg, "content", "")
    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    return {"role": role, "content": str(content or "")}


def build_conversation_context(
    state: dict[str, Any],
    *,
    tail_messages: int = 6,
    exclude_last_user: bool = True,
    max_excerpt_chars: int = 1200,
) -> str:
    """把 rolling_summary / 长期记忆召回 / 最近 N 轮原文拼成一段上下文文本。

    - `tail_messages`: 最近保留多少条原文消息（不含本轮 user 输入）。
    - `exclude_last_user`: 如果 state.messages 末尾是本轮的 HumanMessage，则跳过它
      （本轮 query 由 agent 另行显式拼接）。
    - `max_excerpt_chars`: 每条消息内容的字符上限，防止工具输出撑爆 prompt。
    """

    parts: list[str] = []

    rolling_summary = str(state.get("rolling_summary", "") or "").strip()
    if rolling_summary:
        parts.append("# 历史对话摘要（rolling_summary）\n" + rolling_summary)

    memory_hits: Iterable[Any] = state.get("long_term_memory_hits", []) or []
    hit_lines: list[str] = []
    for hit in memory_hits:
        if isinstance(hit, dict):
            text = str(hit.get("text") or hit.get("summary") or hit.get("content") or "")
        else:
            text = str(hit)
        text = text.strip()
        if text:
            hit_lines.append(f"- {text[:300]}")
    if hit_lines:
        parts.append("# 长期记忆召回\n" + "\n".join(hit_lines[:5]))

    messages_raw = list(state.get("messages", []) or [])
    if messages_raw and exclude_last_user:
        last = messages_raw[-1]
        cls = last.__class__.__name__ if not isinstance(last, dict) else ""
        is_user = cls == "HumanMessage" or (isinstance(last, dict) and last.get("role") == "user")
        if is_user:
            messages_raw = messages_raw[:-1]

    tail = messages_raw[-tail_messages:] if tail_messages > 0 else []
    if tail:
        lines: list[str] = []
        for m in tail:
            d = _msg_to_dict(m)
            content = d["content"]
            if len(content) > max_excerpt_chars:
                content = content[:max_excerpt_chars] + "…(截断)"
            lines.append(f"[{d['role']}] {content}")
        parts.append("# 最近对话原文（最多 {} 条）\n".format(len(tail)) + "\n---\n".join(lines))

    if not parts:
        return ""
    return "\n\n".join(parts)


def serialize_context_for_json(context_text: str) -> str:
    """把上下文文本安全嵌入到 JSON payload 的辅助方法（避免触发意外转义）。"""

    return json.dumps({"context": context_text}, ensure_ascii=False)
