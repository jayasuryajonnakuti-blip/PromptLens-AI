import React, { useState } from 'react'
import { GitBranch, CheckCircle2, XCircle, AlertTriangle, ArrowRight, Clock } from 'lucide-react'
import type { AgentIteration } from '../types/agent'

interface IterationTimelineProps {
  iterations: AgentIteration[]
}

export const IterationTimeline: React.FC<IterationTimelineProps> = ({ iterations }) => {
  const [selectedIdx, setSelectedIdx] = useState(iterations.length - 1)

  if (!iterations || iterations.length === 0) {
    return null
  }

  const activeIteration = iterations[selectedIdx] || iterations[0]

  return (
    <div className="w-full bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-5 sm:p-6 shadow-sm">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-4 border-b border-[var(--border)]">
        <div>
          <div className="flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-indigo-500" />
            <h2 className="text-base font-bold text-[var(--text-primary)]">
              Agent Loop Execution History
            </h2>
          </div>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">
            Bounded optimization with iterative Critic feedback and Validator verification.
          </p>
        </div>

        {/* Iteration Selector Tabs */}
        <div className="flex items-center gap-1.5 bg-[var(--bg-canvas)] p-1 rounded-xl border border-[var(--border)]">
          {iterations.map((it, idx) => {
            const isPass = it.decision === 'PASS'
            const isReview = it.decision === 'NEEDS_REVIEW'
            return (
              <button
                key={it.iteration_number}
                type="button"
                onClick={() => setSelectedIdx(idx)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  selectedIdx === idx
                    ? 'bg-[var(--bg-card)] text-indigo-600 dark:text-indigo-400 shadow-sm border border-[var(--border)]'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                <span>Loop {it.iteration_number}</span>
                {isPass ? (
                  <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                ) : isReview ? (
                  <AlertTriangle className="w-3 h-3 text-amber-500" />
                ) : (
                  <XCircle className="w-3 h-3 text-rose-500" />
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Visual Pipeline Stepper */}
      <div className="mb-5 overflow-x-auto pb-2">
        <div className="flex items-center justify-between min-w-[500px] gap-2 px-2 py-3 rounded-xl bg-[var(--bg-canvas)] border border-[var(--border)]">
          {[
            { label: 'Prompt In', detail: 'Input' },
            { label: 'Analyzer', detail: 'Step 15' },
            { label: 'Optimizer', detail: 'Step 16' },
            { label: 'Critic', detail: 'Step 17' },
            { label: 'Validator', detail: 'Step 18' },
            { label: 'Verdict', detail: activeIteration.decision },
          ].map((stage, idx, arr) => (
            <React.Fragment key={stage.label}>
              <div className="flex flex-col items-center text-center">
                <span className="w-7 h-7 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-500 text-xs font-bold flex items-center justify-center mb-1">
                  {idx + 1}
                </span>
                <span className="text-[11px] font-bold text-[var(--text-primary)]">
                  {stage.label}
                </span>
                <span className="text-[10px] text-[var(--text-muted)] font-mono">
                  {stage.detail}
                </span>
              </div>
              {idx < arr.length - 1 && (
                <ArrowRight className="w-4 h-4 text-[var(--text-muted)] shrink-0" />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Selected Iteration Detail Card */}
      {activeIteration && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Scores & Decision */}
          <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-canvas)]">
            <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)] block mb-2">
              Loop Decision & Scores
            </span>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs text-[var(--text-secondary)]">Verdict:</span>
              <span
                className={`text-xs font-extrabold px-2 py-0.5 rounded-full ${
                  activeIteration.decision === 'PASS'
                    ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20'
                    : activeIteration.decision === 'NEEDS_REVIEW'
                    ? 'bg-amber-500/10 text-amber-500 border border-amber-500/20'
                    : 'bg-rose-500/10 text-rose-500 border border-rose-500/20'
                }`}
              >
                {activeIteration.decision}
              </span>
            </div>

            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="text-[var(--text-secondary)]">Score Before:</span>
              <span className="font-mono font-bold text-[var(--text-primary)]">
                {activeIteration.score_before !== null ? activeIteration.score_before.toFixed(1) : '—'}
              </span>
            </div>

            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="text-[var(--text-secondary)]">Score After:</span>
              <span className="font-mono font-bold text-indigo-500">
                {activeIteration.score_after !== null ? activeIteration.score_after.toFixed(1) : '—'}
              </span>
            </div>

            <div className="flex items-center justify-between text-xs pt-2 border-t border-[var(--border)]">
              <span className="text-[var(--text-muted)] flex items-center gap-1">
                <Clock className="w-3 h-3" /> Latency
              </span>
              <span className="font-mono text-[var(--text-muted)]">
                {Math.round(activeIteration.latency_ms)}ms
              </span>
            </div>
          </div>

          {/* Optimizer Summary */}
          <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-canvas)]">
            <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)] block mb-2">
              Optimizer Modifications
            </span>
            <p className="text-xs text-[var(--text-secondary)] leading-relaxed mb-3">
              {activeIteration.optimizer_result?.summary || 'Applied clarity and constraint improvements.'}
            </p>
            {activeIteration.optimizer_result?.changes && activeIteration.optimizer_result.changes.length > 0 && (
              <div className="flex flex-col gap-1.5">
                {activeIteration.optimizer_result.changes.slice(0, 3).map((ch, cidx) => (
                  <div key={cidx} className="text-[11px] flex items-start gap-1.5 text-[var(--text-secondary)]">
                    <span className="font-bold text-indigo-500 shrink-0 capitalize">• {ch.category}:</span>
                    <span className="truncate">{ch.description}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Critic & Validator Findings */}
          <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-canvas)]">
            <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)] block mb-2">
              Critic & Validator Verdict
            </span>
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="text-[var(--text-secondary)]">Critic Score:</span>
              <span className="font-mono font-bold text-[var(--text-primary)]">
                {activeIteration.critic_result?.overall_critique_score?.toFixed(1) || '—'}
              </span>
            </div>
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="text-[var(--text-secondary)]">Semantic Sim:</span>
              <span className="font-mono font-bold text-[var(--text-primary)]">
                {activeIteration.critic_result?.semantic_similarity
                  ? `${(activeIteration.critic_result.semantic_similarity * 100).toFixed(0)}%`
                  : '—'}
              </span>
            </div>
            <div className="text-[11px] text-[var(--text-muted)] pt-2 border-t border-[var(--border)]">
              {activeIteration.validator_result?.passed_checks?.length || 0} checks passed,{' '}
              {activeIteration.validator_result?.failed_checks?.length || 0} failed.
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
