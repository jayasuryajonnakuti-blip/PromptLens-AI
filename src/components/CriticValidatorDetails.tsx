import React from 'react'
import { CheckCircle, HelpCircle, ShieldAlert, Sparkles, Info } from 'lucide-react'
import type { AgentResponse } from '../types/agent'

interface CriticValidatorDetailsProps {
  response: AgentResponse
}

export const CriticValidatorDetails: React.FC<CriticValidatorDetailsProps> = ({ response }) => {
  const critic = response.final_critic_result
  const validator = response.final_validation

  return (
    <div className="w-full grid grid-cols-1 lg:grid-cols-2 gap-5">
      {/* Critic Evaluation Panel */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-5 sm:p-6 shadow-sm flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between gap-3 mb-4 pb-3 border-b border-[var(--border)]">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-500" />
              <h3 className="text-sm font-bold text-[var(--text-primary)]">
                AI Critic Evaluation (Step 17)
              </h3>
            </div>
            {critic?.decision && (
              <span
                className={`text-[11px] font-bold px-2 py-0.5 rounded-full ${
                  critic.decision === 'PASS'
                    ? 'bg-emerald-500/10 text-emerald-500'
                    : 'bg-amber-500/10 text-amber-500'
                }`}
              >
                {critic.decision}
              </span>
            )}
          </div>

          {critic ? (
            <div className="flex flex-col gap-4">
              {/* Strengths */}
              {critic.strengths && critic.strengths.length > 0 && (
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)] block mb-1.5">
                    Preserved Strengths
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {critic.strengths.map((str, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center gap-1 text-[11px] font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 px-2.5 py-1 rounded-lg border border-emerald-500/20"
                      >
                        <CheckCircle className="w-3 h-3 shrink-0" />
                        {str}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Lost Requirements Warning */}
              {critic.lost_requirements && critic.lost_requirements.length > 0 && (
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-rose-500 block mb-1.5">
                    Lost Requirements Flagged
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {critic.lost_requirements.map((req, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center gap-1 text-[11px] font-medium bg-rose-500/10 text-rose-600 dark:text-rose-400 px-2.5 py-1 rounded-lg border border-rose-500/20"
                      >
                        <ShieldAlert className="w-3 h-3 shrink-0" />
                        {req}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Unsupported Assumptions */}
              {critic.unsupported_assumptions && critic.unsupported_assumptions.length > 0 && (
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-amber-500 block mb-1.5">
                    Unsupported Assumptions
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {critic.unsupported_assumptions.map((asm, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center gap-1 text-[11px] font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 px-2.5 py-1 rounded-lg border border-amber-500/20"
                      >
                        <HelpCircle className="w-3 h-3 shrink-0" />
                        {asm}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Recommendations */}
              {critic.recommendations && critic.recommendations.length > 0 && (
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)] block mb-1.5">
                    Actionable Recommendations
                  </span>
                  <ul className="flex flex-col gap-1.5 text-xs text-[var(--text-secondary)]">
                    {critic.recommendations.map((rec, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-indigo-500 font-bold">•</span>
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <p className="text-xs text-[var(--text-muted)]">
              No Critic feedback recorded for this execution.
            </p>
          )}
        </div>

        <div className="mt-4 pt-3 border-t border-[var(--border)] text-[11px] text-[var(--text-muted)]">
          Semantic Similarity Score:{' '}
          <strong className="text-[var(--text-primary)] font-mono">
            {critic?.semantic_similarity ? `${(critic.semantic_similarity * 100).toFixed(1)}%` : 'N/A'}
          </strong>
        </div>
      </div>

      {/* AI Validator Panel */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-5 sm:p-6 shadow-sm flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between gap-3 mb-4 pb-3 border-b border-[var(--border)]">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-indigo-500" />
              <h3 className="text-sm font-bold text-[var(--text-primary)]">
                AI Validator Verification (Step 18)
              </h3>
            </div>
            {validator?.decision && (
              <span
                className={`text-[11px] font-bold px-2 py-0.5 rounded-full ${
                  validator.decision === 'PASS'
                    ? 'bg-emerald-500/10 text-emerald-500'
                    : 'bg-rose-500/10 text-rose-500'
                }`}
              >
                {validator.decision}
              </span>
            )}
          </div>

          {validator ? (
            <div className="flex flex-col gap-4">
              {/* Passed Checks */}
              {validator.passed_checks && validator.passed_checks.length > 0 && (
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-500 block mb-1.5">
                    Verification Invariants Passed ({validator.passed_checks.length})
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {validator.passed_checks.map((chk, idx) => (
                      <span
                        key={idx}
                        className="text-[10px] font-mono font-medium bg-[var(--bg-canvas)] text-[var(--text-secondary)] px-2 py-0.5 rounded border border-[var(--border)]"
                      >
                        ✓ {chk}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Issues Detected */}
              {validator.issues && validator.issues.length > 0 && (
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-rose-500 block mb-1.5">
                    Validation Violations ({validator.issues.length})
                  </span>
                  <div className="flex flex-col gap-2">
                    {validator.issues.map((iss, idx) => (
                      <div
                        key={idx}
                        className="p-2.5 rounded-lg border border-rose-500/20 bg-rose-500/5 text-xs text-rose-600 dark:text-rose-400"
                      >
                        <div className="flex items-center justify-between font-bold mb-1">
                          <span>{iss.code}</span>
                          <span className="text-[10px] uppercase px-1.5 py-0.2 rounded bg-rose-500/10">
                            {iss.severity}
                          </span>
                        </div>
                        <p className="text-[11px]">{iss.message}</p>
                        {iss.evidence && (
                          <span className="text-[10px] text-[var(--text-muted)] font-mono block mt-1">
                            Evidence: {iss.evidence}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Safety Score */}
              <div className="flex items-center justify-between text-xs py-2 px-3 rounded-xl bg-[var(--bg-canvas)] border border-[var(--border)]">
                <span className="text-[var(--text-secondary)] font-medium">Safety Score:</span>
                <span className="font-mono font-bold text-indigo-500">
                  {validator.safety_score !== undefined ? `${validator.safety_score.toFixed(1)}/100` : '—'}
                </span>
              </div>
            </div>
          ) : (
            <p className="text-xs text-[var(--text-muted)]">
              No detailed Validator records present for this run.
            </p>
          )}
        </div>

        {/* Disclaimers Footer */}
        {response.disclaimers && response.disclaimers.length > 0 && (
          <div className="mt-4 pt-3 border-t border-[var(--border)]">
            <div className="flex items-center gap-1.5 text-[10px] text-[var(--text-muted)]">
              <Info className="w-3 h-3 shrink-0" />
              <span>{response.disclaimers[0]}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
