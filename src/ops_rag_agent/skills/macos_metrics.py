from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import psutil

from ops_rag_agent.observability import trace_skill_call
from ops_rag_agent.skills.base import SkillKind, SkillSpec


@dataclass
class MacosMetricsSkill:
    spec: SkillSpec = SkillSpec(
        skill_id="ops.macos.metrics",
        name="macOS Metrics",
        description=(
            "Collect local macOS CPU/memory/load/swap/top-process metrics "
            "as a JSON snapshot (read-only)."
        ),
        version="1.0.0",
        business_domain="ops",
        kind=SkillKind.REGULAR,
        requires_approval=False,
        is_readonly=True,
        timeout_s=5,
        tags=("ops", "macos", "readonly", "metrics"),
        when_to_use=(
            "需要一次性采集 macOS 本机 CPU/内存/swap/loadavg/Top 进程指标快照时使用。"
            "只读、纯 psutil 实现，适合作为标准诊断入口。"
        ),
        argument_schema={
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Top 进程数量（按 CPU、RSS 排序），范围 3~25，默认 10。",
                    "default": 10,
                }
            },
            "required": [],
        },
        example_invocations=(
            {},
            {"top_n": 5},
        ),
        risk_level="low",
    )

    @trace_skill_call("ops.macos.metrics")
    def invoke(self, arguments: dict[str, Any]) -> str:
        top_n = int(arguments.get("top_n", 10))
        top_n = max(3, min(top_n, 25))

        ts = time.time()
        cpu_count = psutil.cpu_count(logical=True) or 0
        cpu_percent = psutil.cpu_percent(interval=0.25)
        cpu_percents = psutil.cpu_percent(interval=None, percpu=True)
        try:
            load_1, load_5, load_15 = os.getloadavg()
        except OSError:
            load_1, load_5, load_15 = (None, None, None)

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()

        processes: list[dict[str, Any]] = []
        for p in psutil.process_iter(
            attrs=["pid", "name", "username", "cpu_percent", "memory_percent", "memory_info"]
        ):
            info = p.info
            rss = None
            mem_info = info.get("memory_info")
            if mem_info is not None:
                rss = getattr(mem_info, "rss", None)
            processes.append(
                {
                    "pid": info.get("pid"),
                    "name": info.get("name"),
                    "username": info.get("username"),
                    "cpu_percent": info.get("cpu_percent"),
                    "memory_percent": info.get("memory_percent"),
                    "rss_bytes": rss,
                }
            )

        processes.sort(
            key=lambda x: (
                float(x.get("cpu_percent") or 0.0),
                float(x.get("rss_bytes") or 0.0),
            ),
            reverse=True,
        )

        payload: dict[str, Any] = {
            "timestamp": ts,
            "cpu": {
                "logical_cores": cpu_count,
                "percent_total": cpu_percent,
                "percent_per_cpu": cpu_percents,
                "loadavg": {"1m": load_1, "5m": load_5, "15m": load_15},
            },
            "memory": {
                "total_bytes": vm.total,
                "available_bytes": vm.available,
                "used_bytes": vm.used,
                "percent": vm.percent,
            },
            "swap": {
                "total_bytes": swap.total,
                "used_bytes": swap.used,
                "free_bytes": swap.free,
                "percent": swap.percent,
                "sin_bytes": getattr(swap, "sin", None),
                "sout_bytes": getattr(swap, "sout", None),
            },
            "top_processes": processes[:top_n],
        }
        return json.dumps(payload, ensure_ascii=True)

