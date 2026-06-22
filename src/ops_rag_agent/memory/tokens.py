"""Token 统计工具。

- 使用 tiktoken (`cl100k_base`) 做近似 token 计数；对国产/兼容后端（如 qwen、glm、
  claude 等）的 tokenizer 不完全一致，但数量级足够用于预算控制和前端展示。
- 所有函数都带 graceful fallback：tiktoken 不可用时退化为 `len(text) // 2`。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover - environment without tiktoken
    tiktoken = None  # type: ignore


@lru_cache(maxsize=8)
def _get_encoding(name: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str | None, encoding_name: str = "cl100k_base") -> int:
    """统计一段文本的 token 数（失败时退化为字符数 // 2）。"""

    if not text:
        return 0
    enc = _get_encoding(encoding_name)
    if enc is None:
        return max(1, len(text) // 2)
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 2)


def count_message_tokens(
    messages: Iterable[dict[str, Any] | Any],
    encoding_name: str = "cl100k_base",
) -> int:
    """估算一组 {role,content} 消息的总 token 数（含每条消息的 ~4 token 元信息）。"""

    total = 0
    for msg in messages:
        # 兼容 LangChain 消息对象和纯 dict
        if hasattr(msg, "content"):
            content = getattr(msg, "content", "")
        elif isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = str(msg)
        if isinstance(content, list):
            # 有些 LangChain 消息 content 是 list[dict]
            content = "\n".join(
                str(part.get("text", part)) if isinstance(part, dict) else str(part)
                for part in content
            )
        total += count_tokens(str(content), encoding_name) + 4
    return total


def build_usage_snapshot(
    *,
    messages: Iterable[dict[str, Any] | Any] | None = None,
    prompt_text: str | None = None,
    completion_text: str | None = None,
    rolling_summary: str | None = None,
    context_window: int = 200_000,
    encoding_name: str = "cl100k_base",
) -> dict[str, Any]:
    """构造前端/日志展示用的上下文使用量快照。"""

    history_tokens = count_message_tokens(messages or [], encoding_name)
    prompt_tokens = count_tokens(prompt_text, encoding_name) if prompt_text else 0
    completion_tokens = count_tokens(completion_text, encoding_name) if completion_text else 0
    summary_tokens = count_tokens(rolling_summary, encoding_name) if rolling_summary else 0
    total = history_tokens + prompt_tokens + completion_tokens
    percent = (total / context_window * 100.0) if context_window > 0 else 0.0
    return {
        "history_tokens": history_tokens,
        "summary_tokens": summary_tokens,
        "last_prompt_tokens": prompt_tokens,
        "last_completion_tokens": completion_tokens,
        "total": total,
        "context_window": context_window,
        "percent": round(percent, 2),
        "encoding": encoding_name,
    }
