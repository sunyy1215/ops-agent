import importlib
from types import SimpleNamespace

from fastapi.testclient import TestClient

from ops_rag_agent.api.app import create_app

api_module = importlib.import_module("ops_rag_agent.api.app")


class _FakeGraph:
    def invoke(self, state, config=None):
        del state, config
        return {
            "route": "ops",
            "final_answer": "旧结果，不应优先使用",
            "workflow_status": "completed",
            "approval_required": False,
            "approval_status": "not_required",
            "citations": [],
        }

    def get_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id", "cp-approval")
        return SimpleNamespace(
            config={"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}},
            values={
                "route": "ops",
                "final_answer": "等待审批中的新结果",
                "workflow_status": "waiting_approval",
                "approval_required": True,
                "approval_status": "pending",
                "pending_interrupts": [
                    {
                        "interrupt_id": "int-approval",
                        "value": {"type": "approval_required", "payload": {"reason": "needs approval"}},
                    }
                ],
                "ops_hypotheses": [{"id": "h1", "description": "cpu hot"}],
                "ops_recommendations": [{"priority": "P0", "action": "confirm action"}],
                "ops_failed_skills": [{"skill_id": "ops.prometheus.query", "reason": "disabled"}],
            },
            next=("approval_gate",),
            interrupts=(),
            created_at="2026-04-28T00:00:00Z",
            parent_config=None,
            metadata={},
            tasks=(),
        )


def test_invoke_uses_snapshot_state_for_pending_approval(monkeypatch):
    app = create_app()
    client = TestClient(app)

    monkeypatch.setattr(api_module, "build_graph", lambda: _FakeGraph())

    resp = client.post("/invoke", json={"user_query": "cpu alert"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["route"] == "ops"
    assert data["final_answer"] == "等待审批中的新结果"
    assert data["workflow_status"] == "waiting_approval"
    assert data["approval_required"] is True
    assert data["approval_status"] == "pending"
    assert data["pending_interrupts"][0]["interrupt_id"] == "int-approval"
    assert data["ops_hypotheses"][0]["id"] == "h1"
    assert data["ops_recommendations"][0]["priority"] == "P0"
    assert data["ops_failed_skills"][0]["skill_id"] == "ops.prometheus.query"
