from ops_rag_agent.skills.bootstrap import build_skill_registry


def test_registry_registers_remote_skills() -> None:
    registry = build_skill_registry()
    # Existence is enough here; detailed behavior is covered by unit tests for utils/skills.
    registry.get("ops.remote.exec")
    registry.get("ops.remote.snapshot")
    registry.get("ops.remote.service_inspect")


def test_high_frequency_skill_contracts_are_typed() -> None:
    registry = build_skill_registry()
    for skill_id in (
        "rag.search",
        "web.search",
        "ops.local.snapshot",
        "ops.remote.snapshot",
        "ops.prometheus.query",
        "ops.remote.exec",
        "ops.bash.readonly",
        "ops.terminal.exec",
    ):
        manifest = registry.get(skill_id).spec.to_manifest()
        assert manifest["supports_runtime_validation"] is True
        assert manifest["supports_structured_output"] is True
