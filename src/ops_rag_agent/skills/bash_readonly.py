from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ops_rag_agent.observability import trace_skill_call
from ops_rag_agent.skills.base import SkillKind, SkillSpec

ALLOWED_COMMANDS = {
    "awk",
    "cat",
    "df",
    "du",
    "echo",
    "grep",
    "head",
    "hostname",
    "iostat",
    "journalctl",
    "log",
    "lsof",
    "netstat",
    "nproc",
    "pgrep",
    "pmset",
    "ps",
    "sort",
    "sw_vers",
    "sysctl",
    "tail",
    "top",
    "uname",
    "uniq",
    "uptime",
    "vm_stat",
    "wc",
    "whoami",
}

FORBIDDEN_TOKENS = ("&&", "||", ";", ">", "<", ">>", "<<", "$(", "`")


class BashReadonlyArgs(BaseModel):
    cmd: str


class BashReadonlyResult(BaseModel):
    command: str
    argv_segments: list[list[str]] = []
    exit_code: int
    stdout: str = ""
    stderr: str = ""


def _split_pipeline(command: str) -> list[list[str]]:
    # 把违禁 token 显式回传给 LLM，便于它一眼知道改哪里
    hit_tokens = [token for token in FORBIDDEN_TOKENS if token in command]
    if hit_tokens:
        raise ValueError(
            "forbidden_shell_operator: hit="
            + ",".join(hit_tokens)
            + " | 仅允许单条命令或 `|` 管道，禁用 && || ; > < >> << $() ``，"
            "请改写为白名单命令并去掉这些符号后重试"
        )

    segments = [segment.strip() for segment in command.split("|")]
    if not segments or any(not segment for segment in segments):
        raise ValueError("invalid_pipeline: 命令为空或管道两侧有空段")

    argv_segments: list[list[str]] = []
    for segment in segments:
        argv = shlex.split(segment)
        if not argv:
            raise ValueError("empty_command_segment")
        if argv[0] not in ALLOWED_COMMANDS:
            allowed_preview = ", ".join(sorted(ALLOWED_COMMANDS))
            raise ValueError(
                f"command_not_allowed: {argv[0]} 不在白名单。允许的命令: {allowed_preview}"
            )
        argv_segments.append(argv)
    return argv_segments


def _run_pipeline(argv_segments: list[list[str]], timeout_s: int) -> tuple[int, str, str]:
    processes: list[subprocess.Popen[str]] = []
    previous_stdout = None
    try:
        for index, argv in enumerate(argv_segments):
            process = subprocess.Popen(
                argv,
                stdin=previous_stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if previous_stdout is not None:
                previous_stdout.close()
            previous_stdout = process.stdout
            processes.append(process)

        stdout, last_stderr = processes[-1].communicate(timeout=timeout_s)
        # Drain stderr for earlier pipeline segments to avoid losing failure info
        # (e.g. `ps --sort=-%mem` on macOS fails silently while `head` returns 0).
        stderr_parts: list[str] = []
        for i, process in enumerate(processes[:-1]):
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
            seg_stderr = ""
            if process.stderr is not None:
                try:
                    seg_stderr = process.stderr.read() or ""
                except Exception:  # noqa: BLE001
                    seg_stderr = ""
            if seg_stderr:
                stderr_parts.append(f"[seg{i}] {seg_stderr.strip()}")
        if last_stderr:
            stderr_parts.append(f"[seg{len(processes) - 1}] {last_stderr.strip()}")

        # Pick the first non-zero exit code from any pipeline segment.
        # Treat SIGPIPE (-13) as success: it just means a downstream command
        # (e.g. `head`) closed the pipe early after capturing enough output.
        exit_code = 0
        for p in processes:
            rc = p.returncode
            if rc in (0, None, -13):
                continue
            exit_code = rc
            break
        return exit_code, (stdout or "")[-8000:], ("\n".join(stderr_parts))[-8000:]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()


@dataclass
class BashReadonlySkill:
    spec: SkillSpec = SkillSpec(
        skill_id="ops.bash.readonly",
        name="Read-only Bash",
        description="Execute allowlisted local read-only shell commands and simple pipelines.",
        version="1.0.0",
        business_domain="ops",
        kind=SkillKind.REGULAR,
        requires_approval=True,
        is_readonly=True,
        timeout_s=8,
        tags=("ops", "bash", "terminal", "readonly"),
        when_to_use=(
            "需要在本机只读地运行特定 shell 命令排查问题时使用。"
            "适合 ps/vm_stat/df/uptime/lsof/log show 等只读排查；"
            "禁止任何写操作和管道重定向。"
        ),
        argument_schema={
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "完整的 shell 命令字符串（仅允许 ALLOWED_COMMANDS 白名单内的命令与简单管道）。",
                }
            },
            "required": ["cmd"],
        },
        argument_model=BashReadonlyArgs,
        result_model=BashReadonlyResult,
        example_invocations=(
            {"cmd": "ps -Ao pid,comm,%cpu,%mem,rss -m | head -20"},
            {"cmd": "df -h"},
            {"cmd": "uptime"},
        ),
        risk_level="medium",
    )

    @trace_skill_call("ops.bash.readonly")
    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any] | str:
        command = str(arguments.get("cmd", "")).strip()
        if not command:
            return "ERROR: Missing argument: cmd"

        try:
            argv_segments = _split_pipeline(command)
            exit_code, stdout, stderr = _run_pipeline(argv_segments, timeout_s=self.spec.timeout_s)
            payload = {
                "command": command,
                "argv_segments": argv_segments,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
            }
            return payload
        except subprocess.TimeoutExpired:
            return {
                "command": command,
                "argv_segments": [],
                "exit_code": 124,
                "stdout": "",
                "stderr": "command timed out",
            }
        except Exception as exc:  # noqa: BLE001
            return f"SECURITY_ERROR: {type(exc).__name__}: {exc}"
