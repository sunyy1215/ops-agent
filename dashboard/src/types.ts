export interface InvokeRequest {
  user_query?: string
  thread_id?: string
  checkpoint_id?: string
  initial_state?: Record<string, unknown>
  resume?: unknown
}

export interface IntentAnalysis {
  summary?: string
  sub_questions?: string[]
  domain_hints?: string[]
  complexity?: 'simple' | 'moderate' | 'complex' | string
  need_tools?: boolean
  reasoning?: string
}

export interface PlanStep {
  id: string
  title: string
  intent?: string
  suggested_skill_id?: string
  suggested_arguments?: Record<string, unknown>
  expected_output?: string
  status?: 'planned' | 'running' | 'done' | 'failed' | 'skipped' | 'pending_approval' | string
}

export interface ExecutionPlan {
  steps?: PlanStep[]
  reasoning?: string
  created_at?: string
}

export interface PlanProgressEntry {
  step_id?: string
  turn?: number
  skill_id?: string
  status?: string
  observation?: string
  thought?: string
  arguments?: Record<string, unknown>
}

export interface InvokeResponse {
  thread_id: string
  checkpoint_id?: string | null
  route?: string | null
  final_answer: string
  workflow_status: string
  approval_required: boolean
  approval_status?: string | null
  pending_interrupts: Array<Record<string, unknown>>
  citations: string[]
  trace: Record<string, unknown>
  ops_skill_calls?: Array<Record<string, unknown>>
  ops_execution_steps?: Array<Record<string, unknown>>
  ops_hypotheses?: Array<Record<string, unknown>>
  ops_recommendations?: Array<Record<string, unknown>>
  ops_failed_skills?: Array<Record<string, unknown>>
  router_scratchpad?: Array<Record<string, unknown>>
  router_iterations?: number
  router_stop_reason?: string
  intent_analysis?: IntentAnalysis
  execution_plan?: ExecutionPlan
  plan_progress?: PlanProgressEntry[]
  token_usage?: {
    history_tokens?: number
    summary_tokens?: number
    last_prompt_tokens?: number
    last_completion_tokens?: number
    total?: number
    context_window?: number
    percent?: number
    encoding?: string
    [key: string]: unknown
  }
  rolling_summary?: string
}

export interface OpsSkillCallRecord extends Record<string, unknown> {
  turn?: number
  skill_id?: string
  name?: string
  status?: string
  result_status?: string
  duration_ms?: number
  result_excerpt?: string
  result?: string
  arguments?: Record<string, unknown>
}

export interface OpsExecutionStepRecord extends Record<string, unknown> {
  turn?: number
  title?: string
  summary?: string
  command?: string
  skill_id?: string
  status?: string
  duration_ms?: number
  result_excerpt?: string
  risk?: string
}

export interface KbDocument {
  id: string
  text: string
  metadata?: Record<string, unknown>
}

export interface KbIngestRequest {
  collection?: string
  file_path?: string
  directory_path?: string
  id_prefix?: string
  glob?: string
  dry_run?: boolean
  docs?: KbDocument[]
}

export interface KbIngestResponse {
  collection: string
  inserted_count: number
  ids: string[]
  run_id?: string | null
  dry_run: boolean
  stats: Record<string, unknown>
  duplicate_relations: Array<Record<string, string>>
}

export interface KbSearchRequest {
  query: string
  fused_top_k?: number
  rerank_top_k?: number
  visibility?: string
  biz_domain?: string
  owner?: string
  lang?: string
  source?: string
  source_id?: string
  doc_id?: string
  tags?: string[]
}

export interface KbSearchResponse {
  query: string
  rewritten_queries: string[]
  candidate_count: number
  reranked_count: number
  metadata_filters: Record<string, unknown>
  citations: string[]
  results: Array<Record<string, unknown>>
}

export interface MemoryWriteItem {
  memory_id: string
  memory_type: string
  content: string
  user_id?: string
  session_id?: string
  tags?: string[]
  metadata?: Record<string, unknown>
}

export interface MemoryWriteRequest {
  records: MemoryWriteItem[]
}

export interface MemoryWriteResponse {
  written_ids: string[]
  count: number
}

export interface MemoryRecallRequest {
  query: string
  user_id?: string
  top_k?: number
}

export interface MemoryRecallResponse {
  query: string
  count: number
  results: Array<Record<string, unknown>>
}

export interface PublicConfigResponse {
  sections: Record<string, Record<string, unknown>>
  editable_fields: string[]
  sensitive_fields: string[]
}

export interface RuntimeConfigUpdateRequest {
  updates: Record<string, unknown>
}

export interface RuntimeConfigUpdateResponse {
  updated: Record<string, unknown>
  rejected: Array<{ field: string; reason: string }>
  applied_count: number
  public_config: PublicConfigResponse
}

export interface HealthzResponse {
  status: string
  app_name: string
  environment: string
}

export interface TaskStatusSuggestion {
  code: string
  summary: string
  open_action_count: number
  completed_action_count: number
  failed_action_count: number
}

export interface VerificationSummary {
  status: string
  total: number
  passed: number
  failed: number
  pending: number
  latest_result: string
  summary: string
}

export interface SessionSummary {
  thread_id: string
  checkpoint_id?: string | null
  route?: string | null
  workflow_status?: string | null
  approval_required: boolean
  approval_status?: string | null
  updated_at: string
  last_user_query: string
  final_answer: string
  pending_interrupts: Array<Record<string, unknown>>
  task_status_suggestion: TaskStatusSuggestion
  verification_summary: VerificationSummary
  skill_call_counts: {
    planned: number
    executed: number
    failed: number
  }
}

export interface SessionsResponse {
  count: number
  items: SessionSummary[]
}

export interface RunStateResponse {
  thread_id: string
  checkpoint_id?: string | null
  route?: string | null
  workflow_status?: string | null
  approval_required: boolean
  approval_status?: string | null
  next_nodes: string[]
  pending_interrupts: Array<Record<string, unknown>>
  task_status_suggestion: TaskStatusSuggestion
  verification_summary: VerificationSummary
  state: Record<string, unknown>
}

export interface SkillCatalogItem {
  skill_id: string
  kind?: string
  [key: string]: unknown
}

export interface SkillsCatalogResponse {
  allowed_business_domains: string[]
  groups: {
    all: SkillCatalogItem[]
    regular: SkillCatalogItem[]
    complex_dev: SkillCatalogItem[]
    [key: string]: SkillCatalogItem[]
  }
  counts: {
    all: number
    regular: number
    complex_dev: number
  }
}

export interface MetricsSummaryResponse {
  totals: {
    sessions: number
    checkpoints: number
    skills: number
    pending_approvals: number
    executed_skill_calls: number
    failed_skill_calls: number
  }
  workflow_status_counts: Record<string, number>
  route_counts: Record<string, number>
  task_status_suggestion_counts: Record<string, number>
  verification_status_counts: Record<string, number>
  last_updated_at: string
}
