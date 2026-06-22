from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import time
from typing import Any

from pydantic import BaseModel
from pydantic import ValidationError


class RuntimeStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    PENDING_APPROVAL = "pending_approval"


class ValidationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ValidationIssue:
    path: tuple[str, ...] = ()
    code: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": list(self.path),
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class ValidationResult:
    status: ValidationStatus = ValidationStatus.SKIPPED
    model_name: str = ""
    raw_arguments: dict[str, Any] = field(default_factory=dict)
    normalized_arguments: dict[str, Any] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "model_name": self.model_name,
            "raw_arguments": dict(self.raw_arguments),
            "normalized_arguments": dict(self.normalized_arguments),
            "issues": [item.to_dict() for item in self.issues],
        }


@dataclass(frozen=True)
class RuntimeErrorInfo:
    code: str = ""
    message: str = ""
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class RuntimeAuditRecord:
    phase: str
    status: str
    skill_id: str = ""
    timestamp: str = field(default_factory=_utcnow_iso)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status,
            "skill_id": self.skill_id,
            "timestamp": self.timestamp,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class RuntimeResult:
    status: RuntimeStatus
    skill_id: str = ""
    summary: str = ""
    content: str = ""
    structured_output: dict[str, Any] = field(default_factory=dict)
    approval_request: dict[str, Any] = field(default_factory=dict)
    error: RuntimeErrorInfo | None = None
    validation: ValidationResult = field(default_factory=ValidationResult)
    audit: list[RuntimeAuditRecord] = field(default_factory=list)
    raw_output: Any = None

    @property
    def success(self) -> bool:
        return self.status == RuntimeStatus.SUCCESS

    def to_observation_text(self) -> str:
        if self.content:
            return self.content
        if self.summary:
            return self.summary
        if self.error is not None:
            prefix = self.error.code or "runtime_error"
            return f"{prefix}: {self.error.message}"
        return self.status.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "skill_id": self.skill_id,
            "summary": self.summary,
            "content": self.content,
            "structured_output": dict(self.structured_output),
            "approval_request": dict(self.approval_request),
            "error": self.error.to_dict() if self.error is not None else None,
            "validation": self.validation.to_dict(),
            "audit": [item.to_dict() for item in self.audit],
            "raw_output": self.raw_output,
            "success": self.success,
        }


def _runtime_result_payload(result: RuntimeResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, RuntimeResult):
        return result.to_dict()
    return dict(result or {})


def runtime_result_duration_ms(result: RuntimeResult | dict[str, Any]) -> int:
    payload = _runtime_result_payload(result)
    for record in reversed(list(payload.get("audit") or [])):
        details = dict(record.get("details") or {})
        duration = int(details.get("duration_ms") or 0)
        if duration > 0:
            return duration
    return 0


def build_runtime_event(
    *,
    result: RuntimeResult | dict[str, Any],
    turn: int,
    action: str,
    plan_step_id: str = "",
    decision_source: str = "",
) -> dict[str, Any]:
    payload = _runtime_result_payload(result)
    return {
        "turn": turn,
        "action": action,
        "plan_step_id": plan_step_id,
        "decision_source": decision_source,
        "duration_ms": runtime_result_duration_ms(result),
        **payload,
    }


def runtime_result_to_skill_call(
    *,
    result: RuntimeResult | dict[str, Any],
    spec_manifest: dict[str, Any] | None,
    turn: int,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    manifest = dict(spec_manifest or {})
    payload = _runtime_result_payload(result)
    status = str(payload.get("status") or RuntimeStatus.FAILED.value)
    return {
        "turn": turn,
        "skill_id": str(payload.get("skill_id") or ""),
        "arguments": dict(arguments),
        "version": str(manifest.get("version") or ""),
        "business_domain": str(manifest.get("business_domain") or ""),
        "kind": str(manifest.get("kind") or "regular"),
        "requires_approval": bool(manifest.get("requires_approval", False)),
        "status": "done" if status == RuntimeStatus.SUCCESS.value else "failed",
        "result": str(payload.get("content") or payload.get("summary") or "")[:4000],
        "duration_ms": runtime_result_duration_ms(result),
        "success": bool(payload.get("success", False)),
    }


def build_runtime_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    validation_status_counts: dict[str, int] = {}
    by_skill: dict[str, dict[str, Any]] = {}
    total_duration_ms = 0

    for event in events:
        status = str(event.get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        duration_ms = int(event.get("duration_ms") or 0)
        total_duration_ms += duration_ms
        validation = dict(event.get("validation") or {})
        validation_status = str(validation.get("status") or "unknown")
        validation_status_counts[validation_status] = (
            validation_status_counts.get(validation_status, 0) + 1
        )

        skill_id = str(event.get("skill_id") or "")
        if not skill_id:
            continue
        bucket = by_skill.setdefault(
            skill_id,
            {
                "count": 0,
                "success": 0,
                "failed": 0,
                "blocked": 0,
                "pending_approval": 0,
                "total_duration_ms": 0,
            },
        )
        bucket["count"] += 1
        bucket["total_duration_ms"] += duration_ms
        if status == RuntimeStatus.SUCCESS.value:
            bucket["success"] += 1
        elif status == RuntimeStatus.PENDING_APPROVAL.value:
            bucket["pending_approval"] += 1
        elif status == RuntimeStatus.BLOCKED.value:
            bucket["blocked"] += 1
        else:
            bucket["failed"] += 1

    return {
        "total_events": len(events),
        "total_duration_ms": total_duration_ms,
        "status_counts": by_status,
        "validation_status_counts": validation_status_counts,
        "skills": by_skill,
    }


def _model_name(model: type[BaseModel] | None) -> str:
    if model is None:
        return ""
    return f"{model.__module__}.{model.__name__}"


def _issue_from_pydantic_error(item: dict[str, Any]) -> ValidationIssue:
    loc = item.get("loc") or ()
    return ValidationIssue(
        path=tuple(str(part) for part in loc),
        code=str(item.get("type") or "validation_error"),
        message=str(item.get("msg") or "validation error"),
    )


def _is_instance_of_schema_type(value: Any, schema_type: str) -> bool:
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    return True


def validate_arguments(
    *,
    arguments: dict[str, Any],
    argument_model: type[BaseModel] | None = None,
    argument_schema: dict[str, Any] | None = None,
) -> ValidationResult:
    raw_arguments = dict(arguments)
    if argument_model is not None:
        try:
            validated = argument_model.model_validate(raw_arguments)
        except ValidationError as exc:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                model_name=_model_name(argument_model),
                raw_arguments=raw_arguments,
                normalized_arguments={},
                issues=[_issue_from_pydantic_error(item) for item in exc.errors()],
            )
        return ValidationResult(
            status=ValidationStatus.PASSED,
            model_name=_model_name(argument_model),
            raw_arguments=raw_arguments,
            normalized_arguments=validated.model_dump(mode="python"),
        )

    schema = dict(argument_schema or {})
    if not schema:
        return ValidationResult(
            status=ValidationStatus.SKIPPED,
            raw_arguments=raw_arguments,
            normalized_arguments=raw_arguments,
        )

    issues: list[ValidationIssue] = []
    normalized = dict(raw_arguments)
    schema_type = str(schema.get("type") or "object")
    if schema_type == "object":
        if not isinstance(raw_arguments, dict):
            issues.append(
                ValidationIssue(
                    path=(),
                    code="type_error.object",
                    message="arguments must be an object",
                )
            )
            return ValidationResult(
                status=ValidationStatus.FAILED,
                raw_arguments=raw_arguments,
                normalized_arguments={},
                issues=issues,
            )
        properties = dict(schema.get("properties") or {})
        required = [str(item) for item in (schema.get("required") or [])]
        for name in required:
            if name not in normalized:
                issues.append(
                    ValidationIssue(
                        path=(name,),
                        code="missing",
                        message=f"Missing required argument: {name}",
                    )
                )
        for name, property_schema in properties.items():
            property_schema = dict(property_schema or {})
            if name not in normalized and "default" in property_schema:
                normalized[name] = property_schema["default"]
            if name not in normalized:
                continue
            expected_type = str(property_schema.get("type") or "")
            if expected_type and not _is_instance_of_schema_type(normalized[name], expected_type):
                issues.append(
                    ValidationIssue(
                        path=(name,),
                        code=f"type_error.{expected_type}",
                        message=f"Argument '{name}' should be {expected_type}",
                    )
                )
    elif not _is_instance_of_schema_type(raw_arguments, schema_type):
        issues.append(
            ValidationIssue(
                path=(),
                code=f"type_error.{schema_type}",
                message=f"arguments should be {schema_type}",
            )
        )

    return ValidationResult(
        status=ValidationStatus.FAILED if issues else ValidationStatus.PASSED,
        raw_arguments=raw_arguments,
        normalized_arguments={} if issues else normalized,
        issues=issues,
    )


def skill_requires_approval(spec_manifest: dict[str, Any]) -> bool:
    if spec_manifest.get("requires_approval"):
        return True
    risk = str(spec_manifest.get("risk_level") or "low").lower()
    return risk in {"medium", "high"}


def has_completed_rag_search(history: list[dict[str, Any]]) -> bool:
    return any(
        str(item.get("skill_id") or "") == "rag.search"
        for item in history
        if str(item.get("action") or "") in {"call_skill", "approval_pending"}
    )


def detect_duplicate_call(
    history: list[dict[str, Any]],
    *,
    skill_id: str,
    arguments: dict[str, Any],
    window: int,
) -> bool:
    if not skill_id:
        return False
    args_key = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    recent = [
        item
        for item in history[-window:]
        if str(item.get("action") or "") in {"call_skill", "approval_pending"}
    ]
    if len(recent) < window:
        return False
    return all(
        str(item.get("skill_id") or "") == skill_id
        and json.dumps(item.get("arguments") or {}, ensure_ascii=False, sort_keys=True) == args_key
        for item in recent
    )


def _blocked_result(
    *,
    skill_id: str,
    code: str,
    message: str,
    validation: ValidationResult,
    audit: list[RuntimeAuditRecord],
    structured_output: dict[str, Any] | None = None,
    approval_request: dict[str, Any] | None = None,
    status: RuntimeStatus = RuntimeStatus.BLOCKED,
) -> RuntimeResult:
    return RuntimeResult(
        status=status,
        skill_id=skill_id,
        summary=code,
        content=message,
        structured_output=dict(structured_output or {}),
        approval_request=dict(approval_request or {}),
        error=RuntimeErrorInfo(code=code, message=message, retryable=True),
        validation=validation,
        audit=audit,
    )


def _normalize_output(
    raw_output: Any,
    *,
    result_model: type[BaseModel] | None = None,
) -> tuple[str, dict[str, Any]]:
    if result_model is not None:
        payload = raw_output
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {"text": payload}
        validated = result_model.model_validate(payload)
        structured = validated.model_dump(mode="python")
        return json.dumps(structured, ensure_ascii=False), structured

    if isinstance(raw_output, BaseModel):
        structured = raw_output.model_dump(mode="python")
        return json.dumps(structured, ensure_ascii=False), structured
    if isinstance(raw_output, dict):
        return json.dumps(raw_output, ensure_ascii=False), dict(raw_output)
    if isinstance(raw_output, list):
        return json.dumps(raw_output, ensure_ascii=False), {"items": raw_output}

    text = str(raw_output)
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text, {"text": text}
    if isinstance(parsed, dict):
        return text, parsed
    if isinstance(parsed, list):
        return text, {"items": parsed}
    return text, {"value": parsed}


def run_skill_runtime(
    *,
    registry: Any,
    skill_id: str,
    arguments: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    approval_granted: bool = False,
    enforce_rag_first: bool = True,
    duplicate_window: int = 3,
) -> RuntimeResult:
    history = list(history or [])
    raw_arguments = dict(arguments or {})
    audit: list[RuntimeAuditRecord] = []

    try:
        skill = registry.get(skill_id)
    except KeyError:
        return RuntimeResult(
            status=RuntimeStatus.FAILED,
            skill_id=skill_id,
            summary="unknown_skill_id",
            content=f"unknown_skill_id: {skill_id}",
            error=RuntimeErrorInfo(
                code="unknown_skill_id",
                message=f"Unknown skill_id: {skill_id}",
            ),
            validation=ValidationResult(
                status=ValidationStatus.SKIPPED,
                raw_arguments=raw_arguments,
                normalized_arguments=raw_arguments,
            ),
            audit=[
                RuntimeAuditRecord(
                    phase="resolve_skill",
                    status="failed",
                    skill_id=skill_id,
                    details={"arguments": raw_arguments},
                )
            ],
        )

    spec = skill.spec
    spec_manifest = spec.to_manifest()
    audit.append(
        RuntimeAuditRecord(
            phase="resolve_skill",
            status="passed",
            skill_id=skill_id,
            details={"business_domain": spec_manifest.get("business_domain", "")},
        )
    )

    validation = validate_arguments(
        arguments=raw_arguments,
        argument_model=getattr(spec, "argument_model", None),
        argument_schema=getattr(spec, "resolved_argument_schema", lambda: {})(),
    )
    audit.append(
        RuntimeAuditRecord(
            phase="validate",
            status=validation.status.value,
            skill_id=skill_id,
            details={
                "model_name": validation.model_name,
                "issues": [item.to_dict() for item in validation.issues],
            },
        )
    )
    if validation.status == ValidationStatus.FAILED:
        return RuntimeResult(
            status=RuntimeStatus.FAILED,
            skill_id=skill_id,
            summary="invalid_arguments",
            content="invalid_arguments",
            structured_output={
                "expected_argument_schema": spec_manifest.get("argument_schema", {}),
                "issues": [item.to_dict() for item in validation.issues],
            },
            error=RuntimeErrorInfo(
                code="invalid_arguments",
                message="skill arguments validation failed",
                retryable=True,
                details={"issues": [item.to_dict() for item in validation.issues]},
            ),
            validation=validation,
            audit=audit,
        )

    normalized_arguments = dict(validation.normalized_arguments or raw_arguments)
    rag_available = False
    if enforce_rag_first:
        try:
            registry.get("rag.search")
            rag_available = True
        except KeyError:
            rag_available = False

    if (
        enforce_rag_first
        and rag_available
        and not approval_granted
        and skill_id != "rag.search"
        and not has_completed_rag_search(history)
    ):
        audit.append(
            RuntimeAuditRecord(
                phase="policy",
                status="blocked",
                skill_id=skill_id,
                details={"policy": "rag_first", "required_skill_id": "rag.search"},
            )
        )
        return _blocked_result(
            skill_id=skill_id,
            code="rag_search_required",
            message="policy_blocked: rag.search must run before other skills",
            validation=validation,
            audit=audit,
            structured_output={"required_skill_id": "rag.search", "policy": "rag_first"},
        )

    if detect_duplicate_call(
        history,
        skill_id=skill_id,
        arguments=normalized_arguments,
        window=duplicate_window,
    ):
        audit.append(
            RuntimeAuditRecord(
                phase="policy",
                status="blocked",
                skill_id=skill_id,
                details={"policy": "duplicate_window", "window": duplicate_window},
            )
        )
        return _blocked_result(
            skill_id=skill_id,
            code="loop_detected",
            message="detected_loop: same skill+args repeated",
            validation=validation,
            audit=audit,
            structured_output={"policy": "duplicate_window", "window": duplicate_window},
        )

    if not approval_granted and skill_requires_approval(spec_manifest):
        approval_request = {
            "skill_id": skill_id,
            "arguments": normalized_arguments,
            "risk_level": str(spec_manifest.get("risk_level") or "medium"),
            "reason": (
                f"即将调用 {skill_id}，风险等级 "
                f"{spec_manifest.get('risk_level', 'medium')}，需要人工审批。"
            ),
        }
        audit.append(
            RuntimeAuditRecord(
                phase="policy",
                status="pending_approval",
                skill_id=skill_id,
                details=approval_request,
            )
        )
        return _blocked_result(
            skill_id=skill_id,
            code="approval_required",
            message="waiting_for_user_approval",
            validation=validation,
            audit=audit,
            structured_output={"approval_required": True, "risk_level": approval_request["risk_level"]},
            approval_request=approval_request,
            status=RuntimeStatus.PENDING_APPROVAL,
        )

    execute_started_at = time.monotonic()
    try:
        raw_output = skill.invoke(normalized_arguments)
    except Exception as exc:  # noqa: BLE001
        message = f"ERROR: {type(exc).__name__}: {exc}"
        duration_ms = int((time.monotonic() - execute_started_at) * 1000)
        audit.append(
            RuntimeAuditRecord(
                phase="execute",
                status="failed",
                skill_id=skill_id,
                details={"exception_type": type(exc).__name__, "duration_ms": duration_ms},
            )
        )
        return RuntimeResult(
            status=RuntimeStatus.FAILED,
            skill_id=skill_id,
            summary="skill_exception",
            content=message,
            error=RuntimeErrorInfo(
                code="skill_exception",
                message=message,
                details={"exception_type": type(exc).__name__},
            ),
            validation=ValidationResult(
                status=validation.status,
                model_name=validation.model_name,
                raw_arguments=validation.raw_arguments,
                normalized_arguments=normalized_arguments,
                issues=list(validation.issues),
            ),
            audit=audit,
        )

    duration_ms = int((time.monotonic() - execute_started_at) * 1000)
    text_output = str(raw_output)
    if text_output.startswith("SECURITY_ERROR"):
        audit.append(
            RuntimeAuditRecord(
                phase="execute",
                status="blocked",
                skill_id=skill_id,
                details={"raw_output": text_output[:500], "duration_ms": duration_ms},
            )
        )
        return _blocked_result(
            skill_id=skill_id,
            code="security_error",
            message=text_output,
            validation=ValidationResult(
                status=validation.status,
                model_name=validation.model_name,
                raw_arguments=validation.raw_arguments,
                normalized_arguments=normalized_arguments,
                issues=list(validation.issues),
            ),
            audit=audit,
        )
    if text_output.startswith("APPROVAL_REQUIRED"):
        audit.append(
            RuntimeAuditRecord(
                phase="execute",
                status="blocked",
                skill_id=skill_id,
                details={"raw_output": text_output[:500], "duration_ms": duration_ms},
            )
        )
        return _blocked_result(
            skill_id=skill_id,
            code="approval_required_legacy",
            message=text_output,
            validation=ValidationResult(
                status=validation.status,
                model_name=validation.model_name,
                raw_arguments=validation.raw_arguments,
                normalized_arguments=normalized_arguments,
                issues=list(validation.issues),
            ),
            audit=audit,
        )
    if text_output.startswith("ERROR"):
        audit.append(
            RuntimeAuditRecord(
                phase="execute",
                status="failed",
                skill_id=skill_id,
                details={"raw_output": text_output[:500], "duration_ms": duration_ms},
            )
        )
        return RuntimeResult(
            status=RuntimeStatus.FAILED,
            skill_id=skill_id,
            summary="skill_execution_failed",
            content=text_output,
            error=RuntimeErrorInfo(
                code="skill_execution_failed",
                message=text_output,
            ),
            validation=ValidationResult(
                status=validation.status,
                model_name=validation.model_name,
                raw_arguments=validation.raw_arguments,
                normalized_arguments=normalized_arguments,
                issues=list(validation.issues),
            ),
            audit=audit,
            raw_output=raw_output,
        )

    try:
        normalized_text, structured_output = _normalize_output(
            raw_output,
            result_model=getattr(spec, "result_model", None),
        )
    except Exception as exc:  # noqa: BLE001
        audit.append(
            RuntimeAuditRecord(
                phase="normalize",
                status="failed",
                skill_id=skill_id,
                details={"exception_type": type(exc).__name__, "duration_ms": duration_ms},
            )
        )
        return RuntimeResult(
            status=RuntimeStatus.FAILED,
            skill_id=skill_id,
            summary="result_normalization_failed",
            content=f"ERROR: {type(exc).__name__}: {exc}",
            error=RuntimeErrorInfo(
                code="result_normalization_failed",
                message=f"{type(exc).__name__}: {exc}",
                details={"result_model": _model_name(getattr(spec, "result_model", None))},
            ),
            validation=ValidationResult(
                status=validation.status,
                model_name=validation.model_name,
                raw_arguments=validation.raw_arguments,
                normalized_arguments=normalized_arguments,
                issues=list(validation.issues),
            ),
            audit=audit,
            raw_output=raw_output,
        )
    audit.append(
        RuntimeAuditRecord(
            phase="execute",
            status="success",
            skill_id=skill_id,
            details={"raw_type": type(raw_output).__name__, "duration_ms": duration_ms},
        )
    )
    audit.append(
        RuntimeAuditRecord(
            phase="normalize",
            status="success",
            skill_id=skill_id,
            details={"structured_keys": list(structured_output.keys())[:20]},
        )
    )
    return RuntimeResult(
        status=RuntimeStatus.SUCCESS,
        skill_id=skill_id,
        summary=normalized_text[:200],
        content=normalized_text,
        structured_output=structured_output,
        validation=ValidationResult(
            status=validation.status,
            model_name=validation.model_name,
            raw_arguments=validation.raw_arguments,
            normalized_arguments=normalized_arguments,
            issues=list(validation.issues),
        ),
        audit=audit,
        raw_output=raw_output,
    )
