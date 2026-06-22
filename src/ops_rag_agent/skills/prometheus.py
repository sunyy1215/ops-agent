from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx
from pydantic import BaseModel
from pydantic import Field

from ops_rag_agent.config import settings
from ops_rag_agent.observability import trace_skill_call
from ops_rag_agent.skills.base import SkillKind, SkillSpec


class PrometheusQueryArgs(BaseModel):
    query: str


class PrometheusQueryResult(BaseModel):
    status: str
    data: dict[str, Any] = Field(default_factory=dict)
    errorType: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PrometheusQuerySkill:
    spec: SkillSpec = SkillSpec(
        skill_id="ops.prometheus.query",
        name="Prometheus Query",
        description="Run PromQL instant query via Prometheus HTTP API (/api/v1/query).",
        version="1.0.0",
        business_domain="ops",
        kind=SkillKind.REGULAR,
        requires_approval=False,
        is_readonly=True,
        timeout_s=15,
        tags=("ops", "prometheus", "readonly"),
        when_to_use=(
            "需要对接入的 Prometheus 执行 PromQL 即时查询以获取指标值时使用。"
            "只读，适合验证告警表达式或快速查看当前指标数值。"
        ),
        argument_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "PromQL 表达式，例如 up 或 rate(http_requests_total[5m])。",
                }
            },
            "required": ["query"],
        },
        argument_model=PrometheusQueryArgs,
        result_model=PrometheusQueryResult,
        example_invocations=(
            {"query": "up"},
            {"query": "rate(node_cpu_seconds_total{mode=\"user\"}[1m])"},
        ),
        risk_level="low",
    )

    @trace_skill_call("ops.prometheus.query")
    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any] | str:
        if not settings.prometheus_base_url:
            return "Prometheus is not configured: set PROMETHEUS_BASE_URL."

        query = str(arguments.get("query", "")).strip()
        if not query:
            return "Missing argument: query"

        url = settings.prometheus_base_url.rstrip("/") + "/api/v1/query"
        headers: dict[str, str] = {}
        if settings.prometheus_auth_token:
            headers["Authorization"] = f"Bearer {settings.prometheus_auth_token}"

        with httpx.Client(timeout=self.spec.timeout_s) as client:
            resp = client.get(url, params={"query": query}, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        if isinstance(data, dict):
            return data
        return {"status": "error", "data": {}, "error": "unexpected_response_type"}
