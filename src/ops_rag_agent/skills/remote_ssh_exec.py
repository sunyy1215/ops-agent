from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import Field

from ops_rag_agent.config import settings
from ops_rag_agent.observability import trace_skill_call
from ops_rag_agent.skills.base import SkillKind, SkillSpec
from ops_rag_agent.skills.remote_ssh_utils import (
    RemoteCommandPolicy,
    build_ssh_command,
    command_risk,
    is_allowed_target,
)


class RemoteSshExecArgs(BaseModel):
    target_host: str = Field(
        validation_alias=AliasChoices("target_host", "host"),
        serialization_alias="target_host",
    )
    cmd: str
    approved: bool = False


class RemoteSshExecResult(BaseModel):
    target_host: str
    command: str
    approved: bool = False
    risk_level: str = "low"
    exit_code: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class RemoteSshExecSkill:
    spec: SkillSpec = SkillSpec(
        skill_id="ops.remote.exec",
        name="Remote SSH Exec",
        description="Execute a safe, non-interactive command on an allowed remote host via SSH.",
        version="1.0.0",
        business_domain="ops",
        kind=SkillKind.REGULAR,
        requires_approval=False,
        is_readonly=True,
        timeout_s=30,
        tags=("ops", "remote", "ssh"),
        when_to_use=(
            "需要通过 SSH 在白名单主机上执行非交互式只读命令时使用。"
            "受远程命令策略过滤，禁止写/特权操作，复杂命令需人工审批。"
        ),
        argument_schema={
            "type": "object",
            "properties": {
                "target_host": {
                    "type": "string",
                    "description": "目标主机（必须在 allowed_remote_ssh_hosts 白名单内）。",
                },
                "cmd": {
                    "type": "string",
                    "description": "要在远程执行的 shell 命令，必须是只读、非交互式。",
                },
            },
            "required": ["target_host", "cmd"],
        },
        argument_model=RemoteSshExecArgs,
        result_model=RemoteSshExecResult,
        example_invocations=(
            {"target_host": "node-1", "cmd": "uptime"},
            {"target_host": "node-1", "cmd": "df -h"},
        ),
        risk_level="medium",
    )

    @trace_skill_call("ops.remote.exec")
    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any] | str:
        target_host = str(
            arguments.get("target_host", arguments.get("host", settings.remote_ssh_default_host))
        ).strip()
        cmd = str(arguments.get("cmd", "")).strip()
        approved = bool(arguments.get("approved", False))

        if not is_allowed_target(target_host, allowed_hosts=settings.allowed_remote_ssh_hosts):
            return (
                "SECURITY_ERROR: target_host_not_allowed: "
                f"{target_host or '<empty>'}"
            )

        risk, reason = command_risk(cmd, policy=RemoteCommandPolicy())
        if risk == "rejected":
            return f"SECURITY_ERROR: command_rejected: {reason}"
        if risk == "approval_required":
            # Approval is enforced by graph node; still return signal as a safety net.
            if not approved:
                return f"APPROVAL_REQUIRED: {reason}: {cmd}"

        argv = build_ssh_command(
            target_host=target_host,
            port=settings.remote_ssh_port,
            connect_timeout_s=settings.remote_ssh_connect_timeout_s,
            remote_cmd=cmd,
        )

        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=settings.remote_ssh_command_timeout_s,
        )
        out = (completed.stdout or "")[-8000:]
        err = (completed.stderr or "")[-8000:]
        return {
            "target_host": target_host,
            "command": cmd,
            "approved": approved,
            "risk_level": risk,
            "exit_code": completed.returncode,
            "stdout": out,
            "stderr": err,
        }
