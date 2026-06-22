from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable

from langsmith import traceable, tracing_context

from ops_rag_agent.config import settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_langsmith() -> None:
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_tracing_enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_TRACING_V2"] = "true"


def is_langsmith_enabled() -> bool:
    return settings.langsmith_tracing_enabled


def graph_tracing_context(*, thread_id: str, resume: bool) -> Any:
    configure_langsmith()
    return tracing_context(
        project_name=settings.langsmith_project,
        enabled=settings.langsmith_tracing_enabled,
        tags=[settings.environment, "langgraph"],
        metadata={
            "app_name": settings.app_name,
            "graph_name": settings.langgraph_graph_name,
            "thread_id": thread_id,
            "resume": resume,
        },
    )


def trace_graph_node(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        traced = traceable(
            run_type="chain",
            name=name,
            tags=["langgraph", "node"],
        )(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not settings.langsmith_tracing_enabled:
                return func(*args, **kwargs)
            configure_langsmith()
            return traced(*args, **kwargs)

        return wrapper

    return decorator


def trace_skill_call(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        traced = traceable(
            run_type="tool",
            name=name,
            tags=["skill", "tool"],
        )(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not settings.langsmith_tracing_enabled:
                return func(*args, **kwargs)
            configure_langsmith()
            return traced(*args, **kwargs)

        return wrapper

    return decorator


def build_audit_event(
    event_type: str,
    *,
    node: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": utc_now(),
        "event_type": event_type,
        "node": node,
        "details": details or {},
    }


def append_audit_event(
    state: dict[str, Any],
    event_type: str,
    *,
    node: str | None = None,
    details: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    events = list(state.get("audit_trail", []))
    events.append(build_audit_event(event_type, node=node, details=details))
    return events
