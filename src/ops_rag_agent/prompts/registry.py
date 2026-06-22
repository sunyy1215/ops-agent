from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROMPTS_ROOT.parent.parent.parent
AGENT_PERSONA_PATH = REPO_ROOT / "agent.md"

PROMPT_SEMVER_POLICY: dict[str, str] = {
    "major": "Breaking intent or output-contract change.",
    "minor": "Instruction or rubric change without breaking the output contract.",
    "patch": "Copy edits, examples, and metadata-only fixes.",
}

PROMPT_REVIEW_POLICY: tuple[str, ...] = (
    "Semantic prompt changes require semver bumps.",
    "Prompt diffs require peer review before merge.",
    "Matching eval suites must run before release.",
)


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    version: str
    description: str
    relative_path: str
    owner: str
    eval_suite: str
    review_policy: tuple[str, ...] = PROMPT_REVIEW_POLICY

    @property
    def absolute_path(self) -> Path:
        return PROMPTS_ROOT / self.relative_path

    def to_manifest(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "version": self.version,
            "description": self.description,
            "path": str(self.absolute_path),
            "owner": self.owner,
            "eval_suite": self.eval_suite,
            "review_policy": list(self.review_policy),
        }


PROMPT_SPECS: dict[str, PromptSpec] = {
    "supervisor.prepare_input": PromptSpec(
        prompt_id="supervisor.prepare_input",
        version="1.0.0",
        description="Supervisor system prompt for initial routing and safety framing.",
        relative_path="supervisor/v1/system.txt",
        owner="platform-agent",
        eval_suite="evals/dialog/core.jsonl",
    ),
    "supervisor.route_intent": PromptSpec(
        prompt_id="supervisor.route_intent",
        version="1.0.0",
        description="Supervisor routing prompt for choosing dialog, ops, or rag.",
        relative_path="supervisor/route_intent/v1/system.txt",
        owner="platform-agent",
        eval_suite="evals/dialog/core.jsonl",
    ),
    "dialog.planner": PromptSpec(
        prompt_id="dialog.planner",
        version="1.0.0",
        description="Dialog planning prompt for general user assistance.",
        relative_path="dialog/v1/system.txt",
        owner="platform-agent",
        eval_suite="evals/dialog/core.jsonl",
    ),
    "dialog.general": PromptSpec(
        prompt_id="dialog.general",
        version="1.0.0",
        description="General dialog synthesis prompt for non-ops conversations.",
        relative_path="dialog/general/v1/system.txt",
        owner="platform-agent",
        eval_suite="evals/dialog/core.jsonl",
    ),
    "ops.planner": PromptSpec(
        prompt_id="ops.planner",
        version="1.0.0",
        description="Ops planning prompt for diagnosis and approval-aware remediation.",
        relative_path="ops/v1/system.txt",
        owner="platform-agent",
        eval_suite="evals/ops/core.jsonl",
    ),
    "ops.local_health.analyze": PromptSpec(
        prompt_id="ops.local_health.analyze",
        version="1.0.0",
        description="macOS local health analyze prompt for hypotheses JSON generation.",
        relative_path="ops/local_health_analyze/v1/system.txt",
        owner="platform-agent",
        eval_suite="evals/ops/core.jsonl",
    ),
    "ops.local_health.recommend": PromptSpec(
        prompt_id="ops.local_health.recommend",
        version="1.0.0",
        description="macOS local health recommend prompt for recommendations JSON generation.",
        relative_path="ops/local_health_recommend/v1/system.txt",
        owner="platform-agent",
        eval_suite="evals/ops/core.jsonl",
    ),
    "ops.local_health.synthesize": PromptSpec(
        prompt_id="ops.local_health.synthesize",
        version="1.0.0",
        description="macOS local health final report synthesis prompt.",
        relative_path="ops/local_health_synthesize/v1/system.txt",
        owner="platform-agent",
        eval_suite="evals/ops/core.jsonl",
    ),
    "rag.answering": PromptSpec(
        prompt_id="rag.answering",
        version="1.0.0",
        description="Grounded answering prompt for retrieval augmented generation.",
        relative_path="rag/v1/system.txt",
        owner="platform-agent",
        eval_suite="evals/rag/core.jsonl",
    ),
}


def list_prompt_manifests() -> list[dict[str, Any]]:
    return [spec.to_manifest() for spec in PROMPT_SPECS.values()]


def get_prompt_spec(prompt_id: str) -> PromptSpec:
    if prompt_id not in PROMPT_SPECS:
        raise KeyError(f"Unknown prompt_id: {prompt_id}")
    return PROMPT_SPECS[prompt_id]


@lru_cache(maxsize=None)
def load_prompt_text(prompt_id: str) -> str:
    spec = get_prompt_spec(prompt_id)
    return spec.absolute_path.read_text(encoding="utf-8").strip()


def load_agent_persona_text() -> str:
    """Load the repository-level agent persona file without caching.

    The user wants edits to `agent.md` to take effect on the next response
    generation, so this intentionally reads from disk every time.
    """

    if not AGENT_PERSONA_PATH.exists():
        return ""
    return AGENT_PERSONA_PATH.read_text(encoding="utf-8").strip()


def apply_agent_persona(system_prompt: str) -> str:
    persona = load_agent_persona_text()
    if not persona:
        return system_prompt.strip()
    return (
        f"{system_prompt.strip()}\n\n"
        "Additional repository persona instructions from `agent.md`:\n"
        f"{persona}"
    )
