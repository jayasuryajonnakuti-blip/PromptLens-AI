import React from 'react'
import { ShieldCheck, ShieldAlert, AlertTriangle, Zap, CheckCircle2 } from 'lucide-react'
import type { AgentResponse } from '../types/agent'

interface ScoreCardProps {
  response: AgentResponse
}

function getScoreCategory(score: number | null): {
  label: string
  color: string
  bg: string
  border: string
} {
  if (score === null) {
    return { label: 'UNSCORED', color: 'text-slate-500', bg: 'bg-slate-500/10', border: 'border-slate-500/20' }
  }
  if (score >= 90) {
    return { label: 'EXCELLENT', color: 'text-emerald-500', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20' }
  }
  if (score >= 75) {
    return { label: 'STRONG', color: 'text-indigo-500', bg: 'bg-indigo-500/10', border: 'border-indigo-500/20' }
  }
  if (score >= 60) {
    return { label: 'GOOD', color: 'text-blue-500', bg: 'bg-blue-500/10', border: 'border-blue-500/20' }
  }
  if (score >= 40) {
    return { label: 'FAIR', color: 'text-amber-500', bg: 'bg-amber-500/10', border: 'border-amber-500/20' }
  }
  return { label: 'POOR', color: 'text-rose-500', bg: 'bg-rose-500/10', border: 'border-rose-500/20' }
}

export const ScoreCard: React.FC<ScoreCardProps> = ({ response }) => {
  const score = response.final_score
  const category = getScoreCategory(score)
  const scoreDelta = response.metrics?.score_delta

  return (
    <div className="w-full bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-5 sm:p-6 shadow-sm">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 items-center">
        {/* Overall Score Dial / Value */}
        <div className="flex items-center gap-4 border-b md:border-b-0 md:border-r border-[var(--border)] pb-4 md:pb-0 md:pr-4">
          <div
            className={`w-20 h-20 rounded-2xl ${category.bg} border ${category.border} flex flex-col items-center justify-center shrink-0 shadow-inner`}
          >
            <span className={`text-2xl sm:text-3xl font-black ${category.color} tracking-tight`}>
              {score !== null ? Math.round(score) : '—'}
            </span>
            <span className="text-[10px] font-bold text-[var(--text-muted)] tracking-wider">/ 100</span>
          </div>

          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)] block">
              Quality Score
            </span>
            <span className={`text-sm font-extrabold ${category.color} tracking-wide block mt-0.5`}>
              {category.label}
            </span>
            {typeof scoreDelta === 'number' && (
              <span className={`text-xs font-semibold ${scoreDelta >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                {scoreDelta >= 0 ? `+${scoreDelta.toFixed(1)}` : scoreDelta.toFixed(1)} delta
              </span>
            )}
          </div>
        </div>

        {/* Validation Status */}
        <div className="flex flex-col gap-1.5 border-b md:border-b-0 md:border-r border-[var(--border)] pb-4 md:pb-0 md:pr-4">
          <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">
            Validation Status
          </span>
          <div className="flex items-center gap-2">
            {response.is_validated ? (
              <div className="flex items-center gap-1.5 text-emerald-500 font-bold text-sm bg-emerald-500/10 px-2.5 py-1 rounded-lg border border-emerald-500/20">
                <ShieldCheck className="w-4 h-4" />
                <span>VALIDATED</span>
              </div>
            ) : response.status === 'NEEDS_REVIEW' ? (
              <div className="flex items-center gap-1.5 text-amber-500 font-bold text-sm bg-amber-500/10 px-2.5 py-1 rounded-lg border border-amber-500/20">
                <AlertTriangle className="w-4 h-4" />
                <span>NEEDS REVIEW</span>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 text-rose-500 font-bold text-sm bg-rose-500/10 px-2.5 py-1 rounded-lg border border-rose-500/20">
                <ShieldAlert className="w-4 h-4" />
                <span>UNVALIDATED</span>
              </div>
            )}
          </div>
          <span className="text-[11px] text-[var(--text-muted)]">
            Reason: {response.termination_reason.replace(/_/g, ' ')}
          </span>
        </div>

        {/* Agent Loops Executed */}
        <div className="flex flex-col gap-1.5 border-b md:border-b-0 md:border-r border-[var(--border)] pb-4 md:pb-0 md:pr-4">
          <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">
            Agent Iterations
          </span>
          <div className="flex items-baseline gap-2">
            <span className="text-xl font-extrabold text-[var(--text-primary)]">
              {response.iteration_count}
            </span>
            <span className="text-xs text-[var(--text-muted)]">
              of {response.iterations.length || response.iteration_count} completed
            </span>
          </div>
          <div className="flex items-center gap-1 mt-0.5">
            {[1, 2, 3].map((step) => {
              const active = step <= response.iteration_count
              return (
                <div
                  key={step}
                  className={`h-1.5 flex-1 rounded-full ${
                    active ? 'bg-indigo-500' : 'bg-[var(--border)]'
                  }`}
                />
              )
            })}
          </div>
        </div>

        {/* Execution Metrics */}
        <div className="flex flex-col justify-center gap-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-[var(--text-muted)] flex items-center gap-1">
              <Zap className="w-3.5 h-3.5 text-amber-500" />
              Latency
            </span>
            <span className="font-semibold text-[var(--text-primary)] font-mono">
              {response.metrics?.total_latency_ms ? `${Math.round(response.metrics.total_latency_ms)}ms` : '—'}
            </span>
          </div>

          <div className="flex items-center justify-between text-xs">
            <span className="text-[var(--text-muted)]">Expansion</span>
            <span className="font-semibold text-[var(--text-primary)] font-mono">
              {response.metrics?.expansion_ratio ? `${response.metrics.expansion_ratio.toFixed(2)}x` : '1.0x'}
            </span>
          </div>

          {response.persistence_status && (
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--text-muted)]">Persisted</span>
              <span className="font-semibold text-emerald-500 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" />
                SQLite
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
