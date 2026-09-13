export type OptimizationMode = 'balanced' | 'analytical' | 'creative' | 'expert'

export type AgentStatus = 'COMPLETED' | 'FAILED' | 'NEEDS_REVIEW' | 'MAX_ITERATIONS_REACHED' | 'UNAVAILABLE'

export type AgentTerminationReason =
  | 'VALIDATED'
  | 'VALIDATION_FAILED'
  | 'VALIDATION_REVIEW'
  | 'MAX_ITERATIONS'
  | 'SERVICE_UNAVAILABLE'
  | 'ERROR'

export interface OptimizerResultSummary {
  optimized_prompt: string
  summary: string
  changes: Array<{ category: string; description: string }>
  preserved_requirements: Array<{ requirement: string; status: string }>
  placeholders_inserted: string[]
  improvement_score_delta: number
  optimizer_mode: string
}

export interface CriticResultSummary {
  decision: string
  overall_critique_score: number
  issues: Array<{ type?: string; description?: string; severity?: string; message?: string }>
  lost_requirements: string[]
  introduced_requirements: string[]
  unsupported_assumptions: string[]
  strengths: string[]
  weaknesses: string[]
  recommendations: string[]
  semantic_similarity: number
}

export interface ValidationIssue {
  code: string
  severity: string
  message: string
  field_or_scope: string
  evidence: string
}

export interface ValidationResult {
  decision: 'PASS' | 'FAIL' | 'NEEDS_REVIEW'
  is_valid: boolean
  safety_score: number
  original_metadata?: {
    prompt_length: number
    word_count: number
    sentence_count: number
    detected_intent: string
    intent_confidence: number
    quality_score: number
    output_formats?: string[]
  }
  optimized_metadata?: {
    prompt_length: number
    word_count: number
    sentence_count: number
    detected_intent: string
    intent_confidence: number
    quality_score: number
    output_formats?: string[]
  }
  issues: ValidationIssue[]
  passed_checks?: string[]
  failed_checks?: string[]
  metadata?: {
    validator_version: string
    validation_mode: string
    latency_ms: number
    critic_consistency_checked: boolean
  }
}

export interface AgentIteration {
  iteration_number: number
  prompt_before: string
  prompt_after: string
  optimizer_result: OptimizerResultSummary
  critic_result: CriticResultSummary
  validator_result: ValidationResult
  score_before: number | null
  score_after: number | null
  decision: string
  latency_ms: number
}

export interface AgentMetrics {
  total_latency_ms: number
  total_iterations: number
  initial_score: number | null
  final_score: number | null
  score_delta: number | null
  prompt_length_before: number
  prompt_length_after: number
  expansion_ratio: number
  semantic_similarity: number | null
}

export interface AgentResponse {
  original_prompt: string
  final_prompt: string
  status: AgentStatus
  termination_reason: AgentTerminationReason
  iteration_count: number
  is_validated: boolean
  iterations: AgentIteration[]
  final_validation?: ValidationResult | null
  final_critic_result?: CriticResultSummary | null
  final_score: number | null
  metrics: AgentMetrics
  run_id?: string | null
  persistence_status?: string | null
  disclaimers: string[]
}

export interface AgentRunSummary {
  id: string
  created_at: string
  updated_at: string
  original_prompt: string
  final_prompt: string
  status: AgentStatus
  termination_reason: AgentTerminationReason
  is_validated: boolean
  iteration_count: number
  final_score: number | null
  agent_version: string
  mode: OptimizationMode
}

export interface AgentRunDetail extends AgentRunSummary {
  iterations: AgentIteration[]
  score_history: number[]
  critic_result?: CriticResultSummary | null
  final_validation?: ValidationResult | null
  metrics?: AgentMetrics | null
  run_metadata?: { disclaimers?: string[] } | null
}

export interface AgentRunListResponse {
  items: AgentRunSummary[]
  total: number
  skip: number
  limit: number
}
