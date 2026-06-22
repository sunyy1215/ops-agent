from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ops_rag_agent.models.factory import build_chat_llm, build_reasoning_llm
from ops_rag_agent.prompts import apply_agent_persona, load_prompt_text


Priority = Literal["P0", "P1", "P2"]


class LlmHypothesis(BaseModel):
    id: str = Field(min_length=2, max_length=64)
    description: str = Field(min_length=4, max_length=400)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)
    next_checks: list[str] = Field(default_factory=list, max_length=12)


class LlmRecommendation(BaseModel):
    priority: Priority
    action: str = Field(min_length=4, max_length=200)
    rationale: str = Field(min_length=4, max_length=600)
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)
    suggested_commands: list[str] = Field(default_factory=list, max_length=12)


class AnalyzeOutput(BaseModel):
    hypotheses: list[LlmHypothesis] = Field(default_factory=list, max_length=3)


class RecommendOutput(BaseModel):
    recommendations: list[LlmRecommendation] = Field(default_factory=list, max_length=12)


@dataclass(frozen=True)
class LlmRunResult:
    ok: bool
    value: dict[str, Any]
    raw_text: str
    error: str


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}\s*$")


def _extract_json_object(text: str) -> str:
    m = _JSON_OBJECT_RE.search(text.strip())
    return m.group(0) if m else ""


def _dangerous_command(cmd: str) -> bool:
    c = (cmd or "").strip().lower()
    if not c:
        return False
    dangerous = (
        "rm ",
        "sudo rm",
        "kill ",
        "kill -9",
        "pkill",
        "killall",
        "systemctl restart",
        "launchctl",
        "shutdown",
        "reboot",
    )
    return any(tok in c for tok in dangerous)


def _sanitize_recommendations(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in recs:
        rr = dict(r)
        cmds = [str(c) for c in (rr.get("suggested_commands") or []) if str(c).strip()]
        # Keep commands, but add risk hint to rationale if dangerous.
        if any(_dangerous_command(c) for c in cmds):
            rr["rationale"] = (
                str(rr.get("rationale", "")).strip()
                + " (Risk note: the suggested commands may be disruptive; run manually only after understanding impact.)"
            )
        out.append(rr)
    return out


def llm_analyze_local_health(*, evidence: dict[str, Any]) -> LlmRunResult:
    system = apply_agent_persona(load_prompt_text("ops.local_health.analyze"))
    llm = build_reasoning_llm()
    payload = {
        "evidence": evidence,
        "output_schema": {
            "hypotheses": [
                {
                    "id": "string",
                    "description": "string",
                    "confidence": 0.0,
                    "evidence_refs": ["string"],
                    "next_checks": ["string"],
                }
            ]
        },
    }
    prompt = system + "\n\nEVIDENCE_JSON:\n" + json.dumps(payload, ensure_ascii=True)
    raw = ""
    try:
        msg = llm.invoke(prompt)
        raw = getattr(msg, "content", str(msg))
        json_text = _extract_json_object(raw)
        if not json_text:
            raise ValueError("missing_json_object")
        data = json.loads(json_text)
        parsed = AnalyzeOutput.model_validate(data)
        return LlmRunResult(ok=True, value=parsed.model_dump(), raw_text=raw, error="")
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        return LlmRunResult(ok=False, value={}, raw_text=raw, error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        return LlmRunResult(ok=False, value={}, raw_text=raw, error=f"{type(exc).__name__}: {exc}")


def llm_recommend_local_health(*, evidence: dict[str, Any], hypotheses: list[dict[str, Any]]) -> LlmRunResult:
    system = apply_agent_persona(load_prompt_text("ops.local_health.recommend"))
    llm = build_reasoning_llm()
    payload = {
        "evidence": evidence,
        "hypotheses": hypotheses,
        "output_schema": {
            "recommendations": [
                {
                    "priority": "P0",
                    "action": "string",
                    "rationale": "string",
                    "evidence_refs": ["string"],
                    "suggested_commands": ["string"],
                }
            ]
        },
    }
    prompt = system + "\n\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=True)
    raw = ""
    try:
        msg = llm.invoke(prompt)
        raw = getattr(msg, "content", str(msg))
        json_text = _extract_json_object(raw)
        if not json_text:
            raise ValueError("missing_json_object")
        data = json.loads(json_text)
        parsed = RecommendOutput.model_validate(data)
        sanitized = _sanitize_recommendations(parsed.model_dump().get("recommendations", []))
        return LlmRunResult(
            ok=True,
            value={"recommendations": sanitized},
            raw_text=raw,
            error="",
        )
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        return LlmRunResult(ok=False, value={}, raw_text=raw, error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        return LlmRunResult(ok=False, value={}, raw_text=raw, error=f"{type(exc).__name__}: {exc}")


def llm_summarize_local_health(
    *,
    summary: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    context_text: str | None = None,
) -> LlmRunResult:
    system = apply_agent_persona(load_prompt_text("ops.local_health.synthesize"))
    llm = build_chat_llm()
    payload = {
        "summary": summary,
        "hypotheses": hypotheses,
        "recommendations": recommendations,
    }
    parts = [system]
    if context_text:
        parts.append("# 对话上下文（摘要 + 最近原文）\n" + context_text)
    parts.append("INPUT_JSON:\n" + json.dumps(payload, ensure_ascii=True))
    prompt = "\n\n".join(parts)
    raw = ""
    try:
        msg = llm.invoke(prompt)
        raw = getattr(msg, "content", str(msg)).strip()
        if not raw:
            raise ValueError("empty_llm_output")
        return LlmRunResult(ok=True, value={"answer": raw}, raw_text=raw, error="")
    except Exception as exc:  # noqa: BLE001
        return LlmRunResult(ok=False, value={}, raw_text=raw, error=f"{type(exc).__name__}: {exc}")
