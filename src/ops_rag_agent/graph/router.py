from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from ops_rag_agent.models.factory import build_router_llm
from ops_rag_agent.prompts import load_prompt_text

Route = Literal["dialog", "ops", "rag"]


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    routing_reason: dict[str, Any]


def rule_route(
    *,
    query: str,
    ops_keywords: tuple[str, ...],
    rag_keywords: tuple[str, ...],
) -> RouteDecision:
    normalized_query = (query or "").strip().lower()
    if not normalized_query:
        return RouteDecision(
            route="dialog",
            routing_reason={
                "type": "rule_default",
                "reason": "empty_query",
                "matched_keywords": [],
            },
        )

    matched_ops = [keyword for keyword in ops_keywords if keyword and keyword.lower() in normalized_query]
    matched_rag = [keyword for keyword in rag_keywords if keyword and keyword.lower() in normalized_query]

    if matched_ops and not matched_rag:
        return RouteDecision(
            route="ops",
            routing_reason={
                "type": "rule_match",
                "reason": "ops_keywords_matched",
                "matched_keywords": matched_ops,
            },
        )

    if matched_rag and not matched_ops:
        return RouteDecision(
            route="rag",
            routing_reason={
                "type": "rule_match",
                "reason": "rag_keywords_matched",
                "matched_keywords": matched_rag,
            },
        )

    if matched_ops and matched_rag:
        return RouteDecision(
            route="dialog",
            routing_reason={
                "type": "rule_uncertain",
                "reason": "both_ops_and_rag_matched",
                "matched_ops_keywords": matched_ops,
                "matched_rag_keywords": matched_rag,
            },
        )

    return RouteDecision(
        route="dialog",
        routing_reason={
            "type": "rule_uncertain",
            "reason": "no_keywords_matched",
            "matched_keywords": [],
        },
    )


def llm_route(*, query: str, rule_reason: dict[str, Any]) -> RouteDecision:
    system_prompt = load_prompt_text("supervisor.route_intent")
    llm = build_router_llm()
    prompt = (
        f"{system_prompt}\n\n"
        "Return JSON only.\n"
        'Schema: {"route":"dialog|ops|rag","reason":"short reason"}\n\n'
        f"USER_QUERY:\n{query}\n\n"
        f"RULE_REASON:\n{json.dumps(rule_reason, ensure_ascii=False)}"
    )

    try:
        message = llm.invoke(prompt)
        raw = getattr(message, "content", str(message)).strip()
        data = json.loads(raw)
        route = str(data.get("route", "")).strip().lower()
        if route not in ("dialog", "ops", "rag"):
            raise ValueError("invalid_route")

        return RouteDecision(
            route=route,  # type: ignore[return-value]
            routing_reason={
                "type": "llm_route",
                "route": route,
                "reason": str(data.get("reason", "")).strip(),
                "fallback_from": rule_reason,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return RouteDecision(
            route="dialog",
            routing_reason={
                "type": "llm_fallback",
                "reason": f"{type(exc).__name__}: {exc}",
                "fallback_from": rule_reason,
            },
        )
