from ops_rag_agent.agents.ops_agent import run_ops_agent
from ops_rag_agent.config import settings
from ops_rag_agent.ops.subagents import (
    DEFAULT_MAX_EXECUTION_SECONDS,
    DEFAULT_MAX_TOOL_CALLS,
    build_ops_worker_tasks,
    run_worker_task,
)
from ops_rag_agent.skills.bootstrap import build_skill_registry


def test_build_ops_worker_tasks_applies_default_budgets() -> None:
    tasks = build_ops_worker_tasks(["high cpu", "high memory", "disk pressure", "extra"])

    assert len(tasks) == 3
    assert tasks[0]["max_tool_calls"] == DEFAULT_MAX_TOOL_CALLS
    assert tasks[0]["max_execution_seconds"] == DEFAULT_MAX_EXECUTION_SECONDS
    assert tasks[0]["source_priority"][0] == "official_documentation"


def test_run_worker_task_stops_when_time_budget_is_exhausted() -> None:
    worker_task = build_ops_worker_tasks(["high cpu"])[0]
    worker_task["max_execution_seconds"] = 40

    result = run_worker_task(worker_task, build_skill_registry())

    assert result["budget_exhausted"] is True
    assert result["budget_exhausted_reason"] == "execution_time_budget_exceeded"
    assert result["elapsed_seconds"] == 20
    assert [item["source_type"] for item in result["evidence"]] == ["system_metrics"]


def test_run_ops_agent_exposes_budget_and_source_policies() -> None:
    state = {"user_query": "cpu alert with memory pressure"}

    result = run_ops_agent(state, build_skill_registry())

    assert result["ops_execution_target"]["execution_target"] == "local"
    assert result["ops_execution_target"]["target_host"] == ""
    assert result["ops_budget_policy"]["max_tool_calls_per_worker"] == DEFAULT_MAX_TOOL_CALLS
    assert (
        result["ops_budget_policy"]["max_execution_seconds_per_worker"]
        == DEFAULT_MAX_EXECUTION_SECONDS
    )
    assert result["ops_source_policy"]["preferred_sources"] == [
        "official_documentation",
        "monitoring_platform",
        "system_metrics",
        "internal_knowledge_base",
    ]
    assert result["ops_worker_tasks"]
    assert result["ops_worker_tasks"][0]["deprioritized_sources"][0]["source_type"] == "community_forum"
    assert isinstance(result.get("planned_skill_calls", []), list)
    assert isinstance(result.get("executed_skill_calls", []), list)
    assert isinstance(result.get("remediation_actions", []), list)
    assert isinstance(result.get("approval_payload", {}).get("actions", []), list)


def test_run_ops_agent_supports_remote_branch(monkeypatch) -> None:
    # Avoid real ssh in tests.
    import ops_rag_agent.skills.remote_ssh_exec as remote_ssh_exec

    class _DummyCompleted:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = "ok\n"
            self.stderr = ""

    monkeypatch.setattr(remote_ssh_exec.subprocess, "run", lambda *a, **k: _DummyCompleted())

    host = settings.remote_ssh_default_host
    state = {"user_query": f"remote {host} cpu alert"}

    result = run_ops_agent(state, build_skill_registry())

    assert result["ops_execution_target"]["execution_target"] == "remote"
    assert result["ops_execution_target"]["target_host"] == host
    assert any(call.get("skill_id") == "ops.remote.snapshot" for call in result.get("ops_skill_calls", []))
