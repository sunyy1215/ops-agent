from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RemoteCommandPolicy:
    allowed_roots: tuple[str, ...] = (
        "cat",
        "tail",
        "head",
        "grep",
        "egrep",
        "sed",
        "awk",
        "ls",
        "pwd",
        "whoami",
        "id",
        "uname",
        "date",
        "uptime",
        "w",
        "last",
        "ps",
        "top",
        "vmstat",
        "iostat",
        "sar",
        "free",
        "df",
        "du",
        "ss",
        "netstat",
        "lsof",
        "dmesg",
        "journalctl",
        "systemctl",
        "service",
    )
    # High risk actions: "rm -rf", "kill -9", "reboot", "shutdown", etc.
    dangerous_tokens: tuple[str, ...] = (
        "sudo",
        "rm",
        "mv",
        "chmod",
        "chown",
        "kill",
        "pkill",
        "killall",
        "reboot",
        "shutdown",
        "halt",
        "poweroff",
        "dd",
        "mkfs",
        "mount",
        "umount",
        "iptables",
        "nft",
        "firewall-cmd",
        "curl",
        "wget",
    )
    dangerous_substrings: tuple[str, ...] = (
        "|",
        "&&",
        "||",
        ";",
        "$(",
        "`",
        ">",
        ">>",
        "<",
        "2>",
        "/dev/tcp/",
    )


_SSH_TARGET_RE = re.compile(r"^[A-Za-z0-9_.-]+@?[A-Za-z0-9_.:-]+$")


def normalize_ssh_target(target_host: str) -> str:
    value = str(target_host or "").strip()
    if not value:
        return ""
    if not _SSH_TARGET_RE.match(value):
        return ""
    return value


def is_allowed_target(target_host: str, *, allowed_hosts: Iterable[str]) -> bool:
    normalized = normalize_ssh_target(target_host)
    if not normalized:
        return False
    allowed = {normalize_ssh_target(x) for x in allowed_hosts}
    allowed.discard("")
    return normalized in allowed


def command_risk(
    cmd: str, *, policy: RemoteCommandPolicy | None = None
) -> tuple[str, str]:
    """
    Returns: (risk_level, reason) where risk_level is one of:
    - "readonly"
    - "approval_required"
    - "rejected"
    """
    policy = policy or RemoteCommandPolicy()
    raw = str(cmd or "").strip()
    if not raw:
        return ("rejected", "empty_command")

    # Reject obvious shell metacharacters to keep remote exec non-interactive and safe.
    lowered = raw.lower()
    for token in policy.dangerous_substrings:
        if token in lowered:
            return ("rejected", f"shell_metacharacter_detected:{token}")

    try:
        tokens = shlex.split(raw)
    except ValueError:
        return ("rejected", "invalid_shell_syntax")
    if not tokens:
        return ("rejected", "empty_command")

    root = tokens[0].lower()

    # Let dangerous roots surface as approval-required even if they are not in allowlist,
    # so they can be captured into an approval payload rather than silently rejected.
    if root in {x.lower() for x in policy.dangerous_tokens}:
        return ("approval_required", f"dangerous_root:{root}")

    if root not in {x.lower() for x in policy.allowed_roots}:
        return ("rejected", f"command_not_in_allowlist:{root}")

    if root in {"systemctl", "service"}:
        # Read-only checks like `systemctl status` are OK; others require approval.
        verb = (tokens[1].lower() if len(tokens) >= 2 else "").strip()
        if verb and verb not in {"status", "is-active", "is-enabled", "show"}:
            return ("approval_required", f"service_action_requires_approval:{verb}")

    return ("readonly", "allowlisted_readonly")


def build_ssh_command(
    *, target_host: str, port: int, connect_timeout_s: int, remote_cmd: str
) -> list[str]:
    """
    Build an ssh argv list for non-interactive execution.
    """
    normalized = normalize_ssh_target(target_host)
    if not normalized:
        raise ValueError("Invalid target_host")
    if port <= 0 or port > 65535:
        raise ValueError("Invalid port")
    if connect_timeout_s <= 0:
        raise ValueError("Invalid connect_timeout_s")
    if not str(remote_cmd or "").strip():
        raise ValueError("Invalid remote_cmd")

    return [
        "ssh",
        "-p",
        str(int(port)),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={int(connect_timeout_s)}",
        normalized,
        "--",
        remote_cmd,
    ]
