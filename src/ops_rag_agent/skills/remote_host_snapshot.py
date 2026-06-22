from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from ops_rag_agent.observability import trace_skill_call
from ops_rag_agent.skills.base import SkillKind, SkillSpec
from ops_rag_agent.skills.remote_ssh_exec import RemoteSshExecSkill


def _truncate_stdout_lines(result: str, *, max_lines: int) -> str:
    marker = "\nstdout:\n"
    idx = result.find(marker)
    if idx == -1:
        return result
    start = idx + len(marker)
    end_marker = "\nstderr:\n"
    end = result.find(end_marker, start)
    if end == -1:
        return result
    stdout_block = result[start:end]
    lines = stdout_block.splitlines()
    clipped = "\n".join(lines[:max_lines])
    return result[:start] + clipped + result[end:]


class RemoteHostSnapshotArgs(BaseModel):
    target_host: str
    top_n: int = Field(default=8, ge=3, le=20)


class RemoteCommandResult(BaseModel):
    target_host: str = ""
    command: str = ""
    approved: bool = False
    risk_level: str = "low"
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


class RemoteHostSnapshotResult(BaseModel):
    target_host: str
    top_n: int
    sections: dict[str, RemoteCommandResult]
    blocked_section: str = ""


@dataclass
class RemoteHostSnapshotSkill:
    spec: SkillSpec = SkillSpec(
        skill_id="ops.remote.snapshot",
        name="Remote Host Snapshot",
        description=(
            "Collect remote CPU/memory/disk/load/top-process snapshot "
            "through read-only SSH commands."
        ),
        version="1.0.0",
        business_domain="ops",
        kind=SkillKind.REGULAR,
        requires_approval=False,
        is_readonly=True,
        timeout_s=20,
        tags=("ops", "remote", "snapshot", "readonly"),
        when_to_use=(
            "需要通过 SSH 快速采集远程主机 CPU/内存/磁盘/Top 进程快照时使用。"
            "只读组合命令（uptime/free/df/ps），适合作为远程诊断的首步。"
        ),
        argument_schema={
            "type": "object",
            "properties": {
                "target_host": {
                    "type": "string",
                    "description": "目标主机（必须在白名单内）。",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Top 进程条数，范围 3~20，默认 8。",
                    "default": 8,
                },
            },
            "required": ["target_host"],
        },
        argument_model=RemoteHostSnapshotArgs,
        result_model=RemoteHostSnapshotResult,
        example_invocations=(
            {"target_host": "node-1"},
            {"target_host": "node-1", "top_n": 10},
        ),
        risk_level="low",
    )

    @trace_skill_call("ops.remote.snapshot")
    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        target_host = arguments.get("target_host", "")
        top_n = int(arguments.get("top_n", 8))
        top_n = max(3, min(top_n, 20))
        exec_skill = RemoteSshExecSkill()

        commands = {
            "cpu_load": "uptime",
            "memory": "free -m",
            "disk": "df -h",
            # Avoid shell pipes; output is truncated by exec skill anyway.
            "top_processes": "ps -eo pid,comm,%cpu,%mem --sort=-%cpu",
        }
        sections: dict[str, Any] = {}
        blocked_section = ""
        for key, cmd in commands.items():
            res = exec_skill.invoke({"target_host": target_host, "cmd": cmd})
            if isinstance(res, dict):
                payload = dict(res)
            else:
                payload = {
                    "target_host": target_host,
                    "command": cmd,
                    "approved": False,
                    "risk_level": "unknown",
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": str(res),
                }
            if key == "top_processes":
                payload["stdout"] = _truncate_stdout_lines(
                    f"\nstdout:\n{payload.get('stdout', '')}\nstderr:\n{payload.get('stderr', '')}",
                    max_lines=top_n + 1,
                ).split("\nstdout:\n", 1)[-1].split("\nstderr:\n", 1)[0]
            sections[key] = payload
            stderr_text = str(payload.get("stderr") or "")
            if stderr_text.startswith("SECURITY_ERROR") or stderr_text.startswith("APPROVAL_REQUIRED"):
                blocked_section = key
                break
        return {
            "target_host": str(target_host),
            "top_n": top_n,
            "sections": sections,
            "blocked_section": blocked_section,
        }
