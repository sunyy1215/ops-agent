from __future__ import annotations

from typing import Any, Iterable

from ops_rag_agent.skills.base import Skill, SkillKind


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        skill_id = skill.spec.skill_id
        if skill_id in self._skills:
            raise ValueError(f"Duplicate skill_id: {skill_id}")
        if skill.spec.kind == SkillKind.COMPLEX_DEV and not hasattr(skill, "build_subgraph"):
            raise TypeError(
                f"Complex dev skill '{skill_id}' must provide build_subgraph() for sub-graph orchestration."
            )
        self._skills[skill_id] = skill

    def get(self, skill_id: str) -> Skill:
        if skill_id not in self._skills:
            raise KeyError(f"Unknown skill_id: {skill_id}")
        return self._skills[skill_id]

    def list_specs(
        self,
        *,
        kind: SkillKind | str | None = None,
        allowed_business_domains: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        kind_value = kind.value if isinstance(kind, SkillKind) else kind
        allowed_domains = set(allowed_business_domains or [])
        return [
            s.spec.to_manifest()
            for s in self._skills.values()
            if (kind_value is None or s.spec.kind.value == kind_value)
            and (not allowed_domains or s.spec.business_domain in allowed_domains)
        ]

    def grouped_specs(
        self, *, allowed_business_domains: Iterable[str] | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "all": self.list_specs(allowed_business_domains=allowed_business_domains),
            "regular": self.list_specs(
                kind=SkillKind.REGULAR,
                allowed_business_domains=allowed_business_domains,
            ),
            "complex_dev": self.list_specs(
                kind=SkillKind.COMPLEX_DEV,
                allowed_business_domains=allowed_business_domains,
            ),
        }
