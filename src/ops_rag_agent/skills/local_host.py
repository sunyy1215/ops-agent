from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import psutil
from pydantic import BaseModel
from pydantic import Field

from ops_rag_agent.observability import trace_skill_call
from ops_rag_agent.skills.base import SkillKind, SkillSpec


class LocalHostSnapshotArgs(BaseModel):
    top_n: int = Field(default=8, ge=1, le=20)


class ProcessSnapshot(BaseModel):
    pid: Optional[int] = None
    name: Optional[str] = None
    cpu_percent: Optional[Union[float, int]] = None
    memory_percent: Optional[Union[float, int]] = None


class LocalHostSnapshotResult(BaseModel):
    cpu_percent: Union[float, int]
    virtual_memory: dict[str, Any]
    disk_root: dict[str, Any]
    top_processes_by_cpu: list[ProcessSnapshot]


@dataclass
class LocalHostSnapshotSkill:
    spec: SkillSpec = SkillSpec(
        skill_id="ops.local.snapshot",
        name="Local Host Snapshot",
        description="Collect local CPU/mem/disk/net/process snapshot (read-only).",
        version="1.0.0",
        business_domain="ops",
        kind=SkillKind.REGULAR,
        requires_approval=False,
        is_readonly=True,
        timeout_s=5,
        tags=("ops", "localhost", "readonly"),
        when_to_use=(
            "需要用 psutil 采集本机 CPU/内存/根分区/Top 进程快照进行快速诊断时使用。"
            "只读、无外部命令依赖，适合作为首轮排查的基础上下文。"
        ),
        argument_schema={
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "返回按 CPU 排序的 Top 进程数量，默认 8。",
                    "default": 8,
                }
            },
            "required": [],
        },
        argument_model=LocalHostSnapshotArgs,
        result_model=LocalHostSnapshotResult,
        example_invocations=(
            {"top_n": 8},
            {"top_n": 5},
        ),
        risk_level="low",
    )

    @trace_skill_call("ops.local.snapshot")
    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        top_n = int(arguments.get("top_n", 8))
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        cpu = psutil.cpu_percent(interval=0.2)

        procs = []
        for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"]):
            info = p.info
            procs.append(info)
        procs.sort(key=lambda x: (x.get("cpu_percent") or 0), reverse=True)

        payload = {
            "cpu_percent": cpu,
            "virtual_memory": {
                "total": vm.total,
                "available": vm.available,
                "percent": vm.percent,
            },
            "disk_root": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            },
            "top_processes_by_cpu": procs[:top_n],
        }
        return payload
