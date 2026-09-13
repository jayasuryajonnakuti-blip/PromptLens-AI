import React, { useState, useEffect, useCallback } from 'react'
import {
  Clock,
  Search,
  Trash2,
  ExternalLink,
  Loader2,
  AlertCircle,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  RotateCcw,
} from 'lucide-react'
import type { AgentResponse, AgentRunSummary } from '../types/agent'
import { listRuns, getRun, deleteRun } from '../services/api'

interface HistoryViewProps {
  onSelectRun: (response: AgentResponse) => void
}

export const HistoryView: React.FC<HistoryViewProps> = ({ onSelectRun }) => {
  const [runs, setRuns] = useState<AgentRunSummary[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [loadingRunId, setLoadingRunId] = useState<string | null>(null)

  const fetchHistory = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await listRuns(0, 50)
      setRuns(data.items || [])
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to connect to database history'
      setError(msg)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    listRuns(0, 50)
      .then((data) => {
        if (active) {
          setRuns(data.items || [])
          setIsLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (active) {
          const msg = err instanceof Error ? err.message : 'Failed to connect to database history'
          setError(msg)
          setIsLoading(false)
        }
      })
    return () => {
      active = false
    }
  }, [])

  const handleDelete = async (e: React.MouseEvent, runId: string) => {
    e.stopPropagation()
    if (!window.confirm('Delete this saved prompt run from database?')) {
      return
    }
    try {
      await deleteRun(runId)
      setRuns((prev) => prev.filter((r) => r.id !== runId))
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to delete record')
    }
  }

  const handleOpenRun = async (runId: string) => {
    setLoadingRunId(runId)
    try {
      const detail = await getRun(runId)
      // Convert detail to AgentResponse format for workspace loading
      const response: AgentResponse = {
        original_prompt: detail.original_prompt,
        final_prompt: detail.final_prompt,
        status: detail.status,
        termination_reason: detail.termination_reason,
        iteration_count: detail.iteration_count,
        is_validated: detail.is_validated,
        iterations: detail.iterations || [],
        final_critic_result: detail.critic_result || null,
        final_validation: detail.final_validation || null,
        final_score: detail.final_score,
        metrics: detail.metrics || {
          total_latency_ms: 0,
          total_iterations: detail.iteration_count,
          initial_score: null,
          final_score: detail.final_score,
          score_delta: null,
          prompt_length_before: detail.original_prompt.length,
          prompt_length_after: detail.final_prompt.length,
          expansion_ratio: 1.0,
          semantic_similarity: null,
        },
        run_id: detail.id,
        persistence_status: 'persisted',
        disclaimers: detail.run_metadata?.disclaimers || [
          'Optimization is iterative, but bounded.',
          'PASS terminates the loop.',
        ],
      }
      onSelectRun(response)
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to load details')
    } finally {
      setLoadingRunId(null)
    }
  }

  const filteredRuns = runs.filter(
    (r) =>
      r.original_prompt.toLowerCase().includes(searchTerm.toLowerCase()) ||
      r.final_prompt.toLowerCase().includes(searchTerm.toLowerCase()) ||
      r.mode.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const formatDate = (isoStr: string) => {
    try {
      const d = new Date(isoStr)
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    } catch {
      return isoStr
    }
  }

  return (
    <div className="w-full bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-5 sm:p-6 shadow-sm">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 pb-4 border-b border-[var(--border)]">
        <div>
          <div className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-indigo-500" />
            <h2 className="text-lg font-bold text-[var(--text-primary)]">
              Persisted Execution History
            </h2>
          </div>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">
            Structured records stored in SQLite persistence layer (Step 20).
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
            <input
              type="text"
              placeholder="Search prompts..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-8 pr-3 py-1.5 rounded-xl border border-[var(--border)] bg-[var(--bg-canvas)] text-xs text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-1 focus:ring-indigo-500 w-48 sm:w-64"
            />
          </div>

          <button
            type="button"
            onClick={fetchHistory}
            disabled={isLoading}
            className="p-2 rounded-xl border border-[var(--border)] bg-[var(--bg-canvas)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            title="Refresh history"
          >
            <RotateCcw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="mb-4 p-3.5 rounded-xl border border-rose-500/20 bg-rose-500/10 text-rose-600 dark:text-rose-400 flex items-center gap-2 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Loading State */}
      {isLoading ? (
        <div className="py-16 flex flex-col items-center justify-center text-center text-xs text-[var(--text-muted)]">
          <Loader2 className="w-6 h-6 animate-spin text-indigo-500 mb-2" />
          <span>Loading SQLite database records...</span>
        </div>
      ) : filteredRuns.length === 0 ? (
        /* Empty State */
        <div className="py-16 flex flex-col items-center justify-center text-center">
          <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-500 mb-3">
            <Clock className="w-6 h-6" />
          </div>
          <h3 className="text-sm font-bold text-[var(--text-primary)] mb-1">
            {searchTerm ? 'No matching prompts found' : 'No prompt history yet'}
          </h3>
          <p className="text-xs text-[var(--text-muted)] max-w-sm">
            {searchTerm
              ? 'Try another search term or clear the filter.'
              : 'Execute your first prompt in the Workspace to see persistent Agent runs recorded here.'}
          </p>
        </div>
      ) : (
        /* History Grid / List */
        <div className="flex flex-col gap-3">
          {filteredRuns.map((run) => {
            const isOpening = loadingRunId === run.id
            return (
              <div
                key={run.id}
                onClick={() => handleOpenRun(run.id)}
                className="group p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-canvas)] hover:border-indigo-400/50 hover:shadow-sm cursor-pointer transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-4"
              >
                {/* Left: Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1.5">
                    {run.is_validated ? (
                      <span className="flex items-center gap-1 text-[10px] font-bold text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                        <CheckCircle2 className="w-3 h-3" /> PASS
                      </span>
                    ) : run.status === 'NEEDS_REVIEW' ? (
                      <span className="flex items-center gap-1 text-[10px] font-bold text-amber-500 bg-amber-500/10 px-2 py-0.5 rounded-full border border-amber-500/20">
                        <AlertTriangle className="w-3 h-3" /> REVIEW
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-[10px] font-bold text-rose-500 bg-rose-500/10 px-2 py-0.5 rounded-full border border-rose-500/20">
                        <XCircle className="w-3 h-3" /> {run.status}
                      </span>
                    )}

                    <span className="text-[10px] font-bold text-indigo-500 uppercase tracking-wider bg-indigo-500/10 px-2 py-0.5 rounded-full">
                      {run.mode}
                    </span>

                    <span className="text-[11px] text-[var(--text-muted)] font-mono ml-auto sm:ml-0">
                      {formatDate(run.created_at)}
                    </span>
                  </div>

                  <p className="text-xs font-mono text-[var(--text-primary)] font-medium line-clamp-2 leading-relaxed">
                    {run.original_prompt}
                  </p>
                </div>

                {/* Right: Scores & Actions */}
                <div className="flex items-center justify-between sm:justify-end gap-4 shrink-0 pt-2 sm:pt-0 border-t sm:border-t-0 border-[var(--border)]">
                  <div className="text-right">
                    <span className="text-[10px] font-bold text-[var(--text-muted)] uppercase block">
                      Score
                    </span>
                    <span className="text-base font-black text-indigo-600 dark:text-indigo-400 font-mono">
                      {run.final_score !== null ? Math.round(run.final_score) : '—'}
                      <span className="text-[10px] font-normal text-[var(--text-muted)]">/100</span>
                    </span>
                  </div>

                  <div className="text-right hidden sm:block">
                    <span className="text-[10px] font-bold text-[var(--text-muted)] uppercase block">
                      Loops
                    </span>
                    <span className="text-xs font-bold text-[var(--text-secondary)] font-mono">
                      {run.iteration_count}
                    </span>
                  </div>

                  <div className="flex items-center gap-1 pl-2">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleOpenRun(run.id)
                      }}
                      className="p-1.5 rounded-lg text-[var(--text-secondary)] hover:text-indigo-500 hover:bg-[var(--bg-card)] transition-colors"
                      title="Load in workspace"
                    >
                      {isOpening ? (
                        <Loader2 className="w-4 h-4 animate-spin text-indigo-500" />
                      ) : (
                        <ExternalLink className="w-4 h-4" />
                      )}
                    </button>
                    <button
                      type="button"
                      onClick={(e) => handleDelete(e, run.id)}
                      className="p-1.5 rounded-lg text-[var(--text-secondary)] hover:text-rose-500 hover:bg-[var(--bg-card)] transition-colors"
                      title="Delete record"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
