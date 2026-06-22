"""Skill：联网搜索（公网信息）。

实现策略：优先尝试 DuckDuckGo HTML 搜索（无需 API Key），失败则回退到
配置项 `settings.web_search_endpoint` 指向的自定义 SERP 接口。设计成
"无 key 可用"，让 LLM 在需要查实时/公网信息时可以直接调。

返回 JSON：
{
  "query": "...",
  "engine": "duckduckgo" | "custom" | "fallback",
  "results": [
    {"title": "...", "url": "...", "snippet": "..."}
  ]
}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus, unquote
from urllib.request import Request, urlopen

from pydantic import BaseModel
from pydantic import Field

from ops_rag_agent.config import settings
from ops_rag_agent.observability import trace_skill_call
from ops_rag_agent.skills.base import SkillKind, SkillSpec


_DUCKDUCKGO_HTML = "https://duckduckgo.com/html/?q={q}"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# DuckDuckGo HTML 结果块大致格式：
#   <a class="result__a" href="/l/?...uddg=<encoded_url>">Title</a>
#   <a class="result__snippet" ...>snippet</a>
_DDG_RESULT_PATTERN = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_TAG_STRIP = re.compile(r"<[^>]+>")
_DDG_REDIRECT = re.compile(r"uddg=([^&]+)")


class WebSearchArgs(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=10)


class WebSearchItem(BaseModel):
    title: str
    url: str
    snippet: str = ""


class WebSearchResult(BaseModel):
    query: str
    engine: str
    results: list[WebSearchItem] = Field(default_factory=list)
    note: str = ""
    error: str = ""


def _strip_html(text: str) -> str:
    return _TAG_STRIP.sub("", text or "").strip()


def _decode_ddg_url(href: str) -> str:
    match = _DDG_REDIRECT.search(href)
    if not match:
        return href
    try:
        return unquote(match.group(1))
    except Exception:  # noqa: BLE001
        return href


def _search_duckduckgo(query: str, top_k: int, timeout_s: int) -> list[dict[str, str]]:
    url = _DUCKDUCKGO_HTML.format(q=quote_plus(query))
    req = Request(url, headers={"User-Agent": _USER_AGENT, "Accept-Language": "zh-CN,en;q=0.8"})
    with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        html = resp.read().decode("utf-8", errors="ignore")

    out: list[dict[str, str]] = []
    for match in _DDG_RESULT_PATTERN.finditer(html):
        href, title_html, snippet_html = match.groups()
        url_clean = _decode_ddg_url(href)
        title = _strip_html(title_html)[:200]
        snippet = _strip_html(snippet_html)[:400]
        if not url_clean or not title:
            continue
        out.append({"title": title, "url": url_clean, "snippet": snippet})
        if len(out) >= top_k:
            break
    return out


def _search_custom_endpoint(query: str, top_k: int, timeout_s: int) -> list[dict[str, str]]:
    endpoint = (settings.web_search_endpoint or "").strip()
    if not endpoint:
        return []
    full_url = endpoint.replace("{q}", quote_plus(query)).replace("{top_k}", str(top_k))
    req = Request(full_url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        body = resp.read().decode("utf-8", errors="ignore")
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return []
    items = data.get("results") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[dict[str, str]] = []
    for item in items[:top_k]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "title": str(item.get("title") or "")[:200],
                "url": str(item.get("url") or item.get("link") or ""),
                "snippet": str(item.get("snippet") or item.get("description") or "")[:400],
            }
        )
    return out


@dataclass
class WebSearchSkill:
    spec: SkillSpec = SkillSpec(
        skill_id="web.search",
        name="Web Search",
        description=(
            "对公网进行实时搜索，返回 top_k 条标题/URL/摘要。"
            "默认通过 DuckDuckGo HTML 接口（无需 API Key），可选回退到 settings.web_search_endpoint。"
        ),
        version="1.0.0",
        business_domain="general",
        kind=SkillKind.REGULAR,
        requires_approval=False,
        is_readonly=True,
        timeout_s=12,
        tags=("web", "search", "internet", "external"),
        when_to_use=(
            "当用户问题涉及实时/公网信息、最新版本、官方文档原文、第三方资料时使用。"
            "适合：查最新发布、官方 changelog、外部新闻、开源项目用法、公网博客等。"
            "不适合：内部知识库（用 rag.search）、本机系统排查（用 ops.* skills）。"
        ),
        argument_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或自然语言提问，建议短句。",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量，默认 5，最多 10。",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        argument_model=WebSearchArgs,
        result_model=WebSearchResult,
        example_invocations=(
            {"query": "LangGraph interrupt and resume best practices", "top_k": 5},
            {"query": "macOS 26 m5 powermetrics 新增字段"},
        ),
        risk_level="low",
    )

    @trace_skill_call("web.search")
    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return {"query": query, "engine": "fallback", "results": [], "error": "missing_query"}
        top_k = int(arguments.get("top_k") or 5)
        if top_k <= 0 or top_k > 10:
            top_k = 5

        engine = "duckduckgo"
        results: list[dict[str, str]] = []
        try:
            results = _search_duckduckgo(query, top_k=top_k, timeout_s=self.spec.timeout_s)
        except Exception as exc:  # noqa: BLE001
            ddg_error = f"{type(exc).__name__}: {exc}"
            try:
                results = _search_custom_endpoint(query, top_k=top_k, timeout_s=self.spec.timeout_s)
                engine = "custom"
            except Exception as exc2:  # noqa: BLE001
                return {
                    "query": query,
                    "engine": "fallback",
                    "results": [],
                    "error": f"duckduckgo_failed: {ddg_error}; custom_failed: {type(exc2).__name__}: {exc2}",
                }

        if not results:
            return {"query": query, "engine": engine, "results": [], "note": "no_results"}
        return {"query": query, "engine": engine, "results": results}
