from ops_rag_agent.skills.remote_ssh_utils import (
    build_ssh_command,
    command_risk,
    is_allowed_target,
    normalize_ssh_target,
)


def test_normalize_ssh_target_rejects_weird_inputs() -> None:
    assert normalize_ssh_target("") == ""
    assert normalize_ssh_target("   ") == ""
    assert normalize_ssh_target("user@host;rm -rf /") == ""
    assert normalize_ssh_target("user@host|cat /etc/passwd") == ""


def test_is_allowed_target_matches_whitelist() -> None:
    allowed = ("alice@10.0.0.1", "10.0.0.2")
    assert is_allowed_target("alice@10.0.0.1", allowed_hosts=allowed) is True
    assert is_allowed_target("10.0.0.2", allowed_hosts=allowed) is True
    assert is_allowed_target("bob@10.0.0.1", allowed_hosts=allowed) is False


def test_command_risk_allows_readonly_allowlist() -> None:
    risk, _ = command_risk("uptime")
    assert risk == "readonly"

    risk, _ = command_risk("systemctl status nginx --no-pager")
    assert risk == "readonly"


def test_command_risk_requires_approval_for_dangerous_roots() -> None:
    risk, reason = command_risk("rm -rf /tmp/foo")
    assert risk == "approval_required"
    assert "dangerous_root:rm" in reason

    risk, reason = command_risk("sudo reboot")
    assert risk == "approval_required"
    assert "dangerous_root:sudo" in reason


def test_command_risk_rejects_non_allowlisted_commands() -> None:
    risk, reason = command_risk("echo hello")
    assert risk == "rejected"
    assert "command_not_in_allowlist" in reason


def test_command_risk_rejects_shell_metacharacters() -> None:
    risk, reason = command_risk("ps -ef | head")
    assert risk == "rejected"
    assert "shell_metacharacter_detected" in reason


def test_build_ssh_command_builds_noninteractive_argv() -> None:
    argv = build_ssh_command(
        target_host="alice@10.0.0.1",
        port=22,
        connect_timeout_s=5,
        remote_cmd="uptime",
    )
    assert argv[0] == "ssh"
    assert "--" in argv
    assert argv[-1] == "uptime"

