from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


class SkillCall(TypedDict, total=False):
    turn: int
    skill_id: str
    arguments: dict[str, Any]
    version: str
    business_domain: str
    kind: Literal["regular", "complex_dev"]
    requires_approval: bool
    status: Literal["planned", "approved", "done", "failed"]
    result: str
    duration_ms: int
    success: bool


class SkillValidationIssue(TypedDict, total=False):
    path: list[str]
    code: str
    message: str


class SkillValidationResult(TypedDict, total=False):
    status: Literal["passed", "failed", "skipped"]
    model_name: str
    raw_arguments: dict[str, Any]
    normalized_arguments: dict[str, Any]
    issues: list[SkillValidationIssue]


class RuntimeErrorInfo(TypedDict, total=False):
    code: str
    message: str
    retryable: bool
    details: dict[str, Any]


class RuntimeAuditRecord(TypedDict, total=False):
    phase: str
    status: str
    skill_id: str
    timestamp: str
    details: dict[str, Any]


class RuntimeObservation(TypedDict, total=False):
    status: Literal["success", "failed", "blocked", "pending_approval"]
    skill_id: str
    summary: str
    content: str
    structured_output: dict[str, Any]
    error: RuntimeErrorInfo | None
    validation: SkillValidationResult
    audit: list[RuntimeAuditRecord]
    raw_output: Any
    success: bool


class RuntimeEvent(RuntimeObservation, total=False):
    turn: int
    action: str
    plan_step_id: str
    decision_source: str
    duration_ms: int


class SkillManifest(TypedDict, total=False):
    skill_id: str
    name: str
    description: str
    version: str
    business_domain: str
    kind: Literal["regular", "complex_dev"]
    execution_model: Literal["prompt_tool", "subgraph_prompt"]
    lifecycle_policy: str
    git_versioned: bool
    requires_approval: bool
    is_readonly: bool
    timeout_s: int
    tags: list[str]
    when_to_use: str
    argument_schema: dict
    argument_model: str
    result_schema: dict
    result_model: str
    supports_runtime_validation: bool
    supports_structured_output: bool
    example_invocations: list[dict]
    risk_level: str


class RagDocument(TypedDict, total=False):
    doc_id: str
    chunk_id: str
    score: float
    source: str
    text: str
    metadata: dict[str, Any]


class ShortTermMemoryState(TypedDict, total=False):
    active_summary: str
    protected_head_count: int
    protected_tail_count: int
    last_compression_status: Literal["idle", "compressed", "retried", "skipped"]
    last_token_usage: int


class LongTermMemoryRecord(TypedDict, total=False):
    memory_id: str
    memory_type: Literal["preference", "incident", "knowledge_summary"]
    user_id: str
    session_id: str
    content: str
    tags: list[str]
    metadata: dict[str, Any]
    score: float
    source: str


class CompressionReflection(TypedDict, total=False):
    status: Literal["passed", "failed", "skipped"]
    passed: bool
    feedback: list[str]
    missing_sections: list[str]


class OpsWorkerTask(TypedDict, total=False):
    worker_id: str
    title: str
    anomaly_type: str
    execution_target: Literal["local", "remote"]
    target_host: str
    findings: list[str]
    recommendation: str
    max_tool_calls: int
    max_execution_seconds: int
    tool_calls_used: int
    elapsed_seconds: int
    budget_exhausted: bool
    budget_exhausted_reason: str
    evidence: list[dict[str, Any]]
    # Trace of skill-level calls made (or planned but skipped) while executing this task.
    skill_calls: list[SkillCall]
    # Skill failures / blocked executions (e.g., security error, approval-required, runtime error).
    failed_skills: list[dict[str, Any]]
    deprioritized_sources: list[dict[str, Any]]
    source_priority: list[str]


class AuditEvent(TypedDict, total=False):
    timestamp: str
    event_type: str
    node: str | None
    details: dict[str, Any]


class GuardrailEvent(TypedDict, total=False):
    layer: Literal["rule", "model", "tool"]
    status: Literal["passed", "warn", "blocked"]
    action: Literal["allow", "require_approval", "block"]
    reason: str
    details: dict[str, Any]


class InterruptInfo(TypedDict, total=False):
    interrupt_id: str
    value: Any


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    user_query: str
    thread_id: str
    user_id: str
    session_id: str
    rewritten_queries: list[str]
    route: Literal["dialog", "ops", "rag", "end"]
    workflow_status: Literal["running", "waiting_approval", "paused", "completed", "rejected"]
    approval_status: Literal["not_required", "pending", "approved", "rejected"]
    resumable_from: str
    pending_interrupts: list[InterruptInfo]
    audit_trail: list[AuditEvent]
    guardrail_events: list[GuardrailEvent]
    guardrail_status: Literal["passed", "warn", "blocked"]
    guardrail_action: Literal["allow", "require_approval", "block"]
    prompt_versions: dict[str, str]

    protected_head: list[dict[str, Any]]
    protected_tail: list[dict[str, Any]]
    compression_zone: list[dict[str, Any]]
    rolling_summary: str
    short_term_memory: ShortTermMemoryState
    long_term_memory_hits: list[LongTermMemoryRecord]
    long_term_memory_write_queue: list[LongTermMemoryRecord]
    long_term_memory_written_ids: list[str]
    compression_reflection: CompressionReflection
    compression_retry_count: int
    compression_retry_requested: bool
    token_usage: dict[str, int]

    planned_skill_calls: list[SkillCall]
    executed_skill_calls: list[SkillCall]
    available_skills: list[SkillManifest]
    regular_skills: list[SkillManifest]
    complex_skills: list[SkillManifest]
    skill_access_context: dict[str, Any]

    anomaly_list: list[str]
    ops_worker_tasks: list[OpsWorkerTask]
    ops_budget_policy: dict[str, Any]
    ops_source_policy: dict[str, Any]
    remediation_plan: list[str]
    remediation_actions: list[dict[str, Any]]
    ops_evidence: list[dict[str, Any]]
    ops_skill_calls: list[SkillCall]
    ops_execution_steps: list[dict[str, Any]]
    ops_execution_target: dict[str, Any]
    ops_failed_skills: list[dict[str, Any]]
    ops_hypotheses: list[dict[str, Any]]
    ops_recommendations: list[dict[str, Any]]
    macos_powermetrics_authorized: bool
    approval_required: bool
    approval_payload: dict[str, Any]
    execution_results: list[str]
    verification_results: list[str]

    rag_queries: list[str]
    rag_candidates: list[RagDocument]
    rag_reranked: list[RagDocument]
    citations: list[str]

    router_scratchpad: list[dict[str, Any]]
    router_evidence: dict[str, Any]
    router_iterations: int
    router_stop_reason: str
    router_validation_results: list[SkillValidationResult]
    router_runtime_observations: list[RuntimeObservation]
    runtime_audit_records: list[RuntimeAuditRecord]
    runtime_events: list[RuntimeEvent]
    runtime_summary: dict[str, Any]

    # ---- 意图分析 / 规划（用户请求 -> intent -> plan -> execute -> conclude）----
    intent_analysis: dict[str, Any]
    """意图分析结构：{summary, sub_questions[], domain_hints[], complexity,
    need_tools, reasoning}"""

    execution_plan: dict[str, Any]
    """执行计划结构：{steps: [{id, title, intent, suggested_skill_id,
    suggested_arguments, expected_output, status}], created_at,
    reasoning}"""

    plan_progress: list[dict[str, Any]]
    """执行步骤进度：[{step_id, status, observation, skill_id, turn}]"""

    final_answer: str
