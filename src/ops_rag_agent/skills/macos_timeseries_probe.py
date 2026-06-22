from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from ops_rag_agent.observability import trace_skill_call
from ops_rag_agent.skills.base import SkillKind, SkillSpec
from ops_rag_agent.skills.macos_metrics import MacosMetricsSkill
from ops_rag_agent.skills.macos_powermetrics import MacosPowermetricsSkill


@dataclass
class MacosTimeseriesProbeSkill:
    spec: SkillSpec = SkillSpec(
        skill_id="ops.macos.timeseries_probe",
        name="macOS Timeseries Probe",
        description=(
            "Short timeseries probe by sampling metrics every few seconds, with optional "
            "powermetrics at a lower frequency (read-only)."
        ),
        version="1.0.0",
        business_domain="ops",
        kind=SkillKind.REGULAR,
        requires_approval=False,
        is_readonly=True,
        timeout_s=120,
        tags=("ops", "macos", "readonly", "timeseries"),
        when_to_use=(
            "需要在一段时间窗口内周期性采样 macOS 指标以观察趋势/抖动时使用。"
            "只读，可选搭配已审批的 powermetrics 低频采样。"
        ),
        argument_schema={
            "type": "object",
            "properties": {
                "interval_s": {
                    "type": "integer",
                    "description": "相邻 metrics 采样间隔秒数，默认 3。",
                    "default": 3,
                },
                "samples": {
                    "type": "integer",
                    "description": "采样总次数，默认 5。",
                    "default": 5,
                },
            },
            "required": [],
        },
        example_invocations=(
            {"interval_s": 3, "samples": 5},
            {"interval_s": 5, "samples": 10},
        ),
        risk_level="low",
    )

    @trace_skill_call("ops.macos.timeseries_probe")
    def invoke(self, arguments: dict[str, Any]) -> str:
        duration_s = int(arguments.get("duration_s", 60))
        metrics_interval_s = int(arguments.get("metrics_interval_s", 5))
        powermetrics_interval_s = int(arguments.get("powermetrics_interval_s", 15))
        include_powermetrics = bool(arguments.get("include_powermetrics", True))
        powermetrics_approved = bool(arguments.get("powermetrics_approved", False))

        duration_s = max(10, min(duration_s, 180))
        metrics_interval_s = max(1, min(metrics_interval_s, 30))
        powermetrics_interval_s = max(5, min(powermetrics_interval_s, 60))

        metrics_skill = MacosMetricsSkill()
        power_skill = MacosPowermetricsSkill()

        start_ts = time.time()
        end_ts = start_ts + duration_s
        next_metrics = start_ts
        next_power = start_ts

        samples: list[dict[str, Any]] = []
        while time.time() < end_ts:
            now = time.time()
            sample: dict[str, Any] = {"timestamp": now}
            did_any = False

            if now >= next_metrics:
                did_any = True
                next_metrics = now + metrics_interval_s
                try:
                    sample["metrics"] = json.loads(
                        metrics_skill.invoke({"top_n": 5})
                    )
                except Exception as exc:  # noqa: BLE001
                    sample["metrics_error"] = f"{type(exc).__name__}: {exc}"

            if include_powermetrics and now >= next_power:
                next_power = now + powermetrics_interval_s
                if powermetrics_approved:
                    did_any = True
                    try:
                        sample["powermetrics"] = json.loads(
                            power_skill.invoke({"approved": True})
                        )
                    except Exception as exc:  # noqa: BLE001
                        sample["powermetrics_error"] = f"{type(exc).__name__}: {exc}"
                else:
                    sample["powermetrics_skipped"] = "not_approved"

            if did_any:
                samples.append(sample)

            # Sleep lightly to avoid busy looping.
            time.sleep(0.2)

        payload = {
            "timestamp": time.time(),
            "window_s": duration_s,
            "metrics_interval_s": metrics_interval_s,
            "powermetrics_interval_s": powermetrics_interval_s,
            "include_powermetrics": include_powermetrics,
            "powermetrics_approved": powermetrics_approved,
            "sample_count": len(samples),
            "samples": samples,
        }
        return json.dumps(payload, ensure_ascii=True)

