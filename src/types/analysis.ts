export type Timestamp = string

export type Intent =
  | 'CODING'
  | 'EDUCATION'
  | 'RESEARCH'
  | 'WRITING'
  | 'SUMMARIZATION'
  | 'TRANSLATION'
  | 'BUSINESS'
  | 'MARKETING'
  | 'DATA_ANALYSIS'
  | 'CREATIVE'
  | 'IMAGE_GENERATION'
  | 'GENERAL'

export type QualityStatus = 'POOR' | 'FAIR' | 'GOOD' | 'STRONG' | 'EXCELLENT'

export type ValidationStatus = 'PASS' | 'FAIL' | 'WARNING'

export type FindingSeverity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'

export interface ProviderMetadata {
  provider: string
  model: string
  modelVersion?: string
  requestId?: string
}

export interface ProcessingMetadata {
  analysisId: string
  schemaVersion: string
  startedAt: Timestamp
  completedAt?: Timestamp
  durationMs?: number
  providerMetadata?: ProviderMetadata
}

export interface PromptMetadata {
  characterCount: number
  wordCount: number
  lineCount: number
  language?: string
  source?: string
  createdAt?: Timestamp
}

export interface PromptAnalysisRequest {
  prompt: string
  metadata?: PromptMetadata
  requestedAt: Timestamp
  schemaVersion: string
}

export interface PreprocessingResult {
  originalPrompt: string
  normalizedPrompt: string
  detectedLanguage?: string
  segments: string[]
  tokens?: string[]
  transformations: string[]
  metadata: PromptMetadata
}

export interface RuleFinding {
  ruleId: string
  title: string
  description: string
  severity: FindingSeverity
  dimension?: keyof QualityDimensions
  recommendation?: string
}

export interface RuleEngineResult {
  findings: RuleFinding[]
  rulesEvaluated: number
  completedAt: Timestamp
}

export interface NlpFeatureResult {
  language?: string
  keywords: string[]
  entities: string[]
  intents: string[]
  complexityScore?: number
  readabilityScore?: number
  featureValues: Record<string, number | string | boolean>
  completedAt: Timestamp
}

export interface IntentCandidate {
  intent: Intent
  confidence: number
}

export interface IntentClassificationResult {
  intent: Intent
  confidence: number
  alternatives: IntentCandidate[]
  method: string
  completedAt: Timestamp
}

export interface MLQualityPrediction {
  predictedScore: number
  confidence: number
  dimensionScores?: Partial<Record<keyof QualityDimensions, number>>
  modelMetadata: ProviderMetadata
  completedAt: Timestamp
}

export interface EmbeddingMetadata {
  provider: string
  model: string
  dimensions: number
  distanceMetric?: string
}

export interface EmbeddingResult {
  embedding?: number[]
  metadata: EmbeddingMetadata
  completedAt: Timestamp
}

export interface QualityDimension {
  score: number
  status: QualityStatus
  reason: string
  recommendation: string
}

export interface QualityDimensions {
  clarity: QualityDimension
  specificity: QualityDimension
  context: QualityDimension
  goalDefinition: QualityDimension
  constraints: QualityDimension
  outputFormat: QualityDimension
  rolePersona: QualityDimension
  audience: QualityDimension
  ambiguity: QualityDimension
  completeness: QualityDimension
  actionability: QualityDimension
  consistency: QualityDimension
}

export interface OverallQualityScore {
  score: number
  category: QualityStatus
  rationale: string
}

export interface AIAnalyzerResult {
  summary: string
  detectedIssues: string[]
  recommendations: string[]
  dimensions: QualityDimensions
  modelMetadata: ProviderMetadata
  completedAt: Timestamp
}

export interface OptimizationChange {
  area: keyof QualityDimensions | 'general'
  description: string
}

export interface OptimizationResult {
  optimizedPrompt: string
  changes: OptimizationChange[]
  rationale: string
  modelMetadata?: ProviderMetadata
  completedAt: Timestamp
}

export interface CriticResult {
  verdict: string
  strengths: string[]
  issues: string[]
  recommendations: string[]
  score?: number
  modelMetadata?: ProviderMetadata
  completedAt: Timestamp
}

export interface ValidationFinding {
  code: string
  message: string
  status: ValidationStatus
  severity?: FindingSeverity
}

export interface ValidationResult {
  status: ValidationStatus
  findings: ValidationFinding[]
  validatedAt: Timestamp
}

export interface FinalAnalysisResult {
  originalPrompt: string
  detectedIntent: IntentClassificationResult
  overallScore: OverallQualityScore
  qualityCategory: QualityStatus
  dimensions: QualityDimensions
  detectedIssues: string[]
  recommendations: string[]
  preprocessing: PreprocessingResult
  ruleFindings: RuleEngineResult
  nlpFeatures: NlpFeatureResult
  mlPrediction: MLQualityPrediction
  embedding: EmbeddingResult
  aiAnalysis: AIAnalyzerResult
  optimizedPrompt: OptimizationResult
  critic: CriticResult
  validator: ValidationResult
  processingMetadata: ProcessingMetadata
}
