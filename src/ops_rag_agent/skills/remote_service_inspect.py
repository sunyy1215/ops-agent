from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ops_rag_agent.observability import trace_skill_call
from ops_rag_agent.skills.base import SkillKind, SkillSpec
from ops_rag_agent.skills.remote_ssh_exec import RemoteSshExecSkill


def _format_exec_output(result: Any) -> str:
    if isinstance(result, dict):
        return (
            f"target_host={result.get('target_host', '')}\n"
            f"command={result.get('command', '')}\n"
            f"exit_code={result.get('exit_code', '')}\n"
            f"stdout:\n{result.get('stdout', '')}\n"
            f"stderr:\n{result.get('stderr', '')}"
        )
    return str(result)


@dataclass
class RemoteServiceInspectSkill:
    spec: SkillSpec = SkillSpec(
        skill_id="ops.remote.service_inspect",
        name="Remote Service Inspect",
        description="Inspect service status, recent logs, and listening ports on a remote host.",
        version="1.0.0",
        business_domain="ops",
        kind=SkillKind.REGULAR,
        requires_approval=True,
        is_readonly=True,
        timeout_s=30,
        tags=("ops", "remote", "service", "inspect"),
        when_to_use=(
            "需要排查远程主机上某个服务是否存活、看最近日志或端口监听情况时使用。"
            "组合 systemctl status / journalctl / ss，只读但需人工审批。"
        ),
        argument_schema={
            "type": "object",
            "properties": {
                "target_host": {
                    "type": "string",
                    "description": "目标主机（必须在白名单内）。",
                },
                "service_name": {
                    "type": "string",
                    "description": "systemd 服务名，例如 nginx、redis。",
                },
            },
            "required": ["target_host", "service_name"],
        },
        example_invocations=(
            {"target_host": "node-1", "service_name": "nginx"},
            {"target_host": "node-1", "service_name": "redis"},
        ),
        risk_level="medium",
    )

    @trace_skill_call("ops.remote.service_inspect")
    def invoke(self, arguments: dict[str, Any]) -> str:
        target_host = arguments.get("target_host", "")
        service_name = str(arguments.get("service_name", "")).strip()
        port = str(arguments.get("port", "")).strip()
        log_lines = int(arguments.get("log_lines", 120))
        log_lines = max(20, min(log_lines, 400))

        if not service_name and not port:
            return "Missing argument: service_name or port"

        exec_skill = RemoteSshExecSkill()
        parts: list[str] = []

        if service_name:
            # NOTE: `systemctl status` is treated as read-only by remote policy.
            parts.append(
                "[service_status]\n"
                + _format_exec_output(
                    exec_skill.invoke(
                        {
                            "target_host": target_host,
                            "cmd": f"systemctl status {service_name} --no-pager",
                        }
                    )
                )
            )
            parts.append(
                "[recent_logs]\n"
                + _format_exec_output(
                    exec_skill.invoke(
                        {
                            "target_host": target_host,
                            "cmd": f"journalctl -u {service_name} -n {log_lines} --no-pager",
                        }
                    )
                )
            )

        if port:
            # Avoid shell pipes; use `ss` filter expression.
            parts.append(
                "[listening_ports]\n"
                + _format_exec_output(
                    exec_skill.invoke(
                    {
                        "target_host": target_host,
                        "cmd": f"ss -lntp sport = :{port}",
                    }
                )
            )
            )

        return "\n\n".join(parts)
