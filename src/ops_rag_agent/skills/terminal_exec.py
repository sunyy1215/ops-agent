from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from ops_rag_agent.observability import trace_skill_call
from ops_rag_agent.skills.base import SkillKind, SkillSpec


def _looks_privileged(cmd: str) -> bool:
    tokens = shlex.split(cmd)
    if not tokens:
        return False
    privileged = {"sudo", "systemctl", "service", "launchctl", "rm", "mv", "chmod", "chown"}
    return tokens[0] in privileged


class TerminalExecArgs(BaseModel):
    cmd: str
    approved: bool = Field(default=False)


class TerminalExecResult(BaseModel):
    command: str
    approved: bool = False
    privileged: bool = False
    exit_code: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class TerminalExecSkill:
    spec: SkillSpec = SkillSpec(
        skill_id="ops.terminal.exec",
        name="Terminal Exec",
        description="Execute a shell command locally. Privileged commands require approval.",
        version="1.0.0",
        business_domain="ops",
        kind=SkillKind.REGULAR,
        requires_approval=True,
        is_readonly=False,
        timeout_s=30,
        tags=("ops", "terminal", "write"),
        when_to_use=(
            "需要在本机执行非只读 shell 命令（含写/特权操作）时使用。"
            "强制要求人工审批，仅在明确必要且已评估风险时调用。"
        ),
        argument_schema={
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "要在本机执行的 shell 命令字符串，可能包含写/特权操作。",
                },
                "approved": {
                    "type": "boolean",
                    "description": "特权命令时必须显式置为 true 才会执行。",
                    "default": False,
                },
            },
            "required": ["cmd"],
        },
        argument_model=TerminalExecArgs,
        result_model=TerminalExecResult,
        example_invocations=(
            {"cmd": "ls -lh /tmp"},
            {"cmd": "sudo systemctl restart nginx", "approved": True},
        ),
        risk_level="high",
    )

    @trace_skill_call("ops.terminal.exec")
    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any] | str:
        cmd = str(arguments.get("cmd", "")).strip()
        if not cmd:
            return "Missing argument: cmd"

        # NOTE: Approval gating is enforced by graph node. This is a second safety net.
        approved = bool(arguments.get("approved", False))
        privileged = _looks_privileged(cmd)
        if privileged and not approved:
            return f"APPROVAL_REQUIRED: privileged command detected: {cmd}"

        completed = subprocess.run(
            cmd,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.spec.timeout_s,
        )
        out = (completed.stdout or "")[-8000:]
        err = (completed.stderr or "")[-8000:]
        return {
            "command": cmd,
            "approved": approved,
            "privileged": privileged,
            "exit_code": completed.returncode,
            "stdout": out,
            "stderr": err,
        }
