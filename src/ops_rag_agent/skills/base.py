from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel


class SkillKind(str, Enum):
    REGULAR = "regular"
    COMPLEX_DEV = "complex_dev"


class SkillExecutionModel(str, Enum):
    PROMPT_TOOL = "prompt_tool"
    SUBGRAPH_PROMPT = "subgraph_prompt"


@dataclass(frozen=True)
class SkillKindProfile:
    kind: SkillKind
    execution_model: SkillExecutionModel
    lifecycle_policy: str
    git_versioned: bool = True


SKILL_KIND_PROFILES: dict[SkillKind, SkillKindProfile] = {
    SkillKind.REGULAR: SkillKindProfile(
        kind=SkillKind.REGULAR,
        execution_model=SkillExecutionModel.PROMPT_TOOL,
        lifecycle_policy="Prompt-injected tool skill for direct planning and invocation.",
    ),
    SkillKind.COMPLEX_DEV: SkillKindProfile(
        kind=SkillKind.COMPLEX_DEV,
        execution_model=SkillExecutionModel.SUBGRAPH_PROMPT,
        lifecycle_policy="LangGraph sub-graph skill for multi-step automation with prompt orchestration.",
    ),
}


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    name: str
    description: str
    version: str = "1.0.0"
    business_domain: str = "general"
    kind: SkillKind = SkillKind.REGULAR
    requires_approval: bool = False
    is_readonly: bool = True
    timeout_s: int = 30
    tags: tuple[str, ...] = ()
    # ---- ReAct 路由元数据 ----
    # 何时使用这个 skill：给 LLM 看的「选择提示」
    when_to_use: str = ""
    # 参数 schema（JSON Schema 风格的简化版）：让 LLM 知道参数如何传
    argument_schema: dict[str, Any] = field(default_factory=dict)
    # 强类型参数模型入口，Task 2 的 runtime 校验会优先消费它
    argument_model: type[BaseModel] | None = None
    # 结果 schema / 模型入口：为统一结构化 observation 预留契约
    result_schema: dict[str, Any] = field(default_factory=dict)
    result_model: type[BaseModel] | None = None
    # 典型调用示例：few-shot 给 LLM 参考
    example_invocations: tuple[dict[str, Any], ...] = ()
    # 风险等级：low / medium / high，默认按 requires_approval 决定
    risk_level: str = "low"

    def resolved_argument_schema(self) -> dict[str, Any]:
        if self.argument_schema:
            return dict(self.argument_schema)
        if self.argument_model is None:
            return {}
        return dict(self.argument_model.model_json_schema(mode="validation"))

    def resolved_result_schema(self) -> dict[str, Any]:
        if self.result_schema:
            return dict(self.result_schema)
        if self.result_model is None:
            return {}
        return dict(self.result_model.model_json_schema(mode="serialization"))

    def to_manifest(self) -> dict[str, Any]:
        profile = SKILL_KIND_PROFILES[self.kind]
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "business_domain": self.business_domain,
            "kind": self.kind.value,
            "execution_model": profile.execution_model.value,
            "lifecycle_policy": profile.lifecycle_policy,
            "git_versioned": profile.git_versioned,
            "requires_approval": self.requires_approval,
            "is_readonly": self.is_readonly,
            "timeout_s": self.timeout_s,
            "tags": list(self.tags),
            "when_to_use": self.when_to_use,
            "argument_schema": self.resolved_argument_schema(),
            "argument_model": (
                f"{self.argument_model.__module__}.{self.argument_model.__name__}"
                if self.argument_model is not None
                else ""
            ),
            "result_schema": self.resolved_result_schema(),
            "result_model": (
                f"{self.result_model.__module__}.{self.result_model.__name__}"
                if self.result_model is not None
                else ""
            ),
            "supports_runtime_validation": self.argument_model is not None
            or bool(self.argument_schema),
            "supports_structured_output": self.result_model is not None
            or bool(self.result_schema),
            "example_invocations": [dict(x) for x in self.example_invocations],
            "risk_level": self.risk_level,
        }


class Skill(Protocol):
    spec: SkillSpec

    def invoke(self, arguments: dict[str, Any]) -> Any: ...


class ComplexSkill(Protocol):
    spec: SkillSpec

    def build_subgraph(self) -> Any: ...
