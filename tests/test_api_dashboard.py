from __future__ import annotations

import importlib
from types import SimpleNamespace

from fastapi.testclient import TestClient

from ops_rag_agent.api.app import create_app

api_module = importlib.import_module("ops_rag_agent.api.app")


class _FakeCheckpointer:
    def __init__(self, snapshots: dict[tuple[str, str], SimpleNamespace], history: dict[str, list[SimpleNamespace]]) -> None:
        self._snapshots = snapshots
        self._history = history

    def get_tuple(self, config: dict) -> SimpleNamespace | None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        if checkpoint_id:
            snapshot = self._snapshots.get((thread_id, checkpoint_id))
            if snapshot is None:
                return None
            return SimpleNamespace(config=snapshot.config)

        thread_history = self._history.get(thread_id, [])
        if not thread_history:
            return None
        return SimpleNamespace(config=thread_history[0].config)

    def list(self, config: dict | None):
        del config
        for thread_history in self._history.values():
            for snapshot in thread_history:
                yield SimpleNamespace(config=snapshot.config)


class _FakeGraph:
    def __init__(self, snapshots: dict[tuple[str, str], SimpleNamespace], history: dict[str, list[SimpleNamespace]]) -> None:
        self._snapshots = snapshots
        self._history = history

    def get_state(self, config: dict, *, subgraphs: bool = False) -> SimpleNamespace:
        del subgraphs
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        if checkpoint_id:
            return self._snapshots[(thread_id, checkpoint_id)]
        return self._history[thread_id][0]

    def get_state_history(self, config: dict, *, filter=None, before=None, limit: int | None = None):
        del filter, before
        thread_id = config["configurable"]["thread_id"]
        items = self._history[thread_id]
        if limit is None:
            return iter(items)
        return iter(items[:limit])


class _FakeRegistry:
    def grouped_specs(self, *, allowed_business_domains=None) -> dict[str, list[dict[str, object]]]:
        del allowed_business_domains
        return {
            "all": [
                {"skill_id": "ops.remote.exec", "kind": "regular"},
                {"skill_id": "ops.review.plan", "kind": "complex_dev"},
            ],
            "regular": [{"skill_id": "ops.remote.exec", "kind": "regular"}],
            "complex_dev": [{"skill_id": "ops.review.plan", "kind": "complex_dev"}],
        }


def _make_snapshot(
    *,
    thread_id: str,
    checkpoint_id: str,
    created_at: str,
    values: dict,
    next_nodes: tuple[str, ...] = (),
    parent_checkpoint_id: str | None = None,
    interrupts: tuple[object, ...] = (),
) -> SimpleNamespace:
    parent_config = None
    if parent_checkpoint_id:
        parent_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": parent_checkpoint_id,
            }
        }
    return SimpleNamespace(
        values=values,
        next=next_nodes,
        config={"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}},
        metadata={},
        created_at=created_at,
        parent_config=parent_config,
        tasks=(),
        interrupts=interrupts,
    )


def test_config_public_and_runtime_update(monkeypatch) -> None:
    app = create_app()
    client = TestClient(app)

    monkeypatch.setattr(api_module.settings, "llm_api_key", "secret-value")
    monkeypatch.setattr(api_module.settings, "llm_model_chat", "gpt-old")

    public_resp = client.get("/config/public")
    assert public_resp.status_code == 200
    public_data = public_resp.json()
    assert public_data["sections"]["secrets"]["llm_api_key_configured"] is True
    assert "secret-value" not in public_resp.text

    update_resp = client.put(
        "/config/runtime",
        json={
            "updates": {
                "llm_model_chat": "gpt-4.1-mini",
                "llm_api_key": "new-secret",
                "unknown_field": "x",
            }
        },
    )
    assert update_resp.status_code == 200
    data = update_resp.json()
    assert data["updated"]["llm_model_chat"] == "gpt-4.1-mini"
    assert api_module.settings.llm_model_chat == "gpt-4.1-mini"
    rejected = {item["field"]: item["reason"] for item in data["rejected"]}
    assert "llm_api_key" in rejected
    assert "unknown_field" in rejected


def test_dashboard_management_endpoints(monkeypatch) -> None:
    app = create_app()
    client = TestClient(app)

    thread1_latest = _make_snapshot(
        thread_id="thread-1",
        checkpoint_id="cp-2",
        created_at="2026-04-28T10:00:00Z",
        next_nodes=("approval_gate",),
        interrupts=(SimpleNamespace(id="int-1", value={"type": "approval_required"}),),
        parent_checkpoint_id="cp-1",
        values={
            "user_query": "cpu alert",
            "route": "ops",
            "workflow_status": "waiting_approval",
            "approval_required": True,
            "approval_status": "pending",
            "final_answer": "Awaiting approval",
            "remediation_actions": [{"action_id": "a-1", "status": "planned"}],
            "planned_skill_calls": [{"skill_id": "ops.remote.exec"}],
            "executed_skill_calls": [
                {"skill_id": "ops.remote.exec", "status": "done"},
                {"skill_id": "ops.remote.exec", "status": "failed"},
            ],
            "runtime_events": [
                {"skill_id": "ops.remote.exec", "status": "success", "duration_ms": 12, "validation": {"status": "passed"}},
                {"skill_id": "ops.remote.exec", "status": "failed", "duration_ms": 8, "validation": {"status": "failed"}},
            ],
            "runtime_summary": {
                "total_events": 2,
                "status_counts": {"success": 1, "failed": 1},
            },
            "execution_results": ["action_id=a-1\nstatus=done"],
            "verification_results": [],
        },
    )
    thread1_older = _make_snapshot(
        thread_id="thread-1",
        checkpoint_id="cp-1",
        created_at="2026-04-28T09:30:00Z",
        values={
            "user_query": "cpu alert",
            "route": "ops",
            "workflow_status": "running",
            "approval_required": False,
            "approval_status": "not_required",
            "final_answer": "Investigating",
            "verification_results": [],
        },
    )
    thread2_latest = _make_snapshot(
        thread_id="thread-2",
        checkpoint_id="cp-9",
        created_at="2026-04-28T09:00:00Z",
        values={
            "user_query": "check service",
            "route": "ops",
            "workflow_status": "completed",
            "approval_required": False,
            "approval_status": "approved",
            "final_answer": "Done",
            "remediation_actions": [{"action_id": "a-2", "status": "done"}],
            "planned_skill_calls": [{"skill_id": "ops.review.plan"}],
            "executed_skill_calls": [{"skill_id": "ops.remote.exec", "status": "done"}],
            "runtime_events": [
                {"skill_id": "ops.remote.exec", "status": "success", "duration_ms": 20, "validation": {"status": "passed"}}
            ],
            "runtime_summary": {
                "total_events": 1,
                "status_counts": {"success": 1},
            },
            "execution_results": ["action_id=a-2\nstatus=done"],
            "verification_results": ["service health check passed"],
        },
    )

    history = {
        "thread-1": [thread1_latest, thread1_older],
        "thread-2": [thread2_latest],
    }
    snapshots = {
        ("thread-1", "cp-2"): thread1_latest,
        ("thread-1", "cp-1"): thread1_older,
        ("thread-2", "cp-9"): thread2_latest,
    }

    monkeypatch.setattr(api_module, "build_checkpointer", lambda: _FakeCheckpointer(snapshots, history))
    monkeypatch.setattr(api_module, "build_graph", lambda: _FakeGraph(snapshots, history))
    monkeypatch.setattr(api_module, "build_skill_registry", lambda: _FakeRegistry())

    sessions_resp = client.get("/sessions")
    assert sessions_resp.status_code == 200
    sessions_data = sessions_resp.json()
    assert sessions_data["count"] == 2
    assert sessions_data["items"][0]["thread_id"] == "thread-1"
    assert sessions_data["items"][0]["task_status_suggestion"]["code"] == "awaiting_approval"
    assert sessions_data["items"][0]["verification_summary"]["status"] == "pending"

    session_resp = client.get("/sessions/thread-1")
    assert session_resp.status_code == 200
    session_data = session_resp.json()
    assert session_data["session"]["checkpoint_id"] == "cp-2"
    assert session_data["count"] == 2
    assert session_data["checkpoints"][0]["parent_checkpoint_id"] == "cp-1"

    run_state_resp = client.get("/runs/thread-1/state")
    assert run_state_resp.status_code == 200
    run_state = run_state_resp.json()
    assert run_state["checkpoint_id"] == "cp-2"
    assert run_state["pending_interrupts"][0]["interrupt_id"] == "int-1"
    assert run_state["task_status_suggestion"]["code"] == "awaiting_approval"

    skills_resp = client.get("/skills/catalog")
    assert skills_resp.status_code == 200
    skills_data = skills_resp.json()
    assert skills_data["counts"] == {"all": 2, "regular": 1, "complex_dev": 1}

    metrics_resp = client.get("/metrics/summary")
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert metrics["totals"]["sessions"] == 2
    assert metrics["totals"]["checkpoints"] == 3
    assert metrics["totals"]["skills"] == 2
    assert metrics["workflow_status_counts"]["waiting_approval"] == 1
    assert metrics["workflow_status_counts"]["completed"] == 1
    assert metrics["verification_status_counts"]["pending"] == 1
    assert metrics["verification_status_counts"]["passed"] == 1

    runtime_resp = client.get("/runtime/summary")
    assert runtime_resp.status_code == 200
    runtime_data = runtime_resp.json()
    assert runtime_data["totals"]["sessions"] == 2
    assert runtime_data["totals"]["runtime_events"] == 3
    assert runtime_data["status_counts"]["success"] == 2
    assert runtime_data["status_counts"]["failed"] == 1
