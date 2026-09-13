import React from 'react'
import { Sparkles, Trash2, ArrowRight, Loader2, AlertCircle } from 'lucide-react'
import type { OptimizationMode } from '../types/agent'
import { EXAMPLE_PROMPTS } from '../data/examplePrompts'

interface PromptEditorProps {
  prompt: string
  setPrompt: (value: string) => void
  mode: OptimizationMode
  setMode: (mode: OptimizationMode) => void
  maxIterations: number
  setMaxIterations: (iters: number) => void
  onSubmit: () => void
  isLoading: boolean
  error: string | null
  setError: (err: string | null) => void
}

const MODES: Array<{ id: OptimizationMode; label: string; desc: string }> = [
  { id: 'balanced', label: 'Balanced', desc: 'Clarity, structure & intent preservation' },
  { id: 'analytical', label: 'Analytical', desc: 'Rigorous logic, edge-cases & constraints' },
  { id: 'creative', label: 'Creative', desc: 'Engaging style, tone & descriptive nuance' },
  { id: 'expert', label: 'Expert', desc: 'Domain depth, precision & professional jargon' },
]

export const PromptEditor: React.FC<PromptEditorProps> = ({
  prompt,
  setPrompt,
  mode,
  setMode,
  maxIterations,
  setMaxIterations,
  onSubmit,
  isLoading,
  error,
  setError,
}) => {
  const wordCount = prompt.trim() ? prompt.trim().split(/\s+/).length : 0
  const charCount = prompt.length

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault()
      if (prompt.trim() && !isLoading) {
        onSubmit()
      }
    }
  }

  return (
    <div className="w-full bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-5 sm:p-6 shadow-sm">
      {/* Error Alert */}
      {error && (
        <div className="mb-4 p-3.5 rounded-xl border border-rose-500/20 bg-rose-500/10 text-rose-600 dark:text-rose-400 flex items-start justify-between gap-3 text-xs leading-relaxed">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={() => setError(null)}
            className="text-rose-500 hover:text-rose-700 font-bold px-1"
          >
            ✕
          </button>
        </div>
      )}

      {/* Editor Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <label htmlFor="prompt-input" className="text-xs font-bold uppercase tracking-wider text-[var(--text-secondary)]">
          Original Prompt
        </label>
        <div className="flex items-center gap-3 text-xs text-[var(--text-muted)] font-mono">
          <span>{wordCount} words</span>
          <span>•</span>
          <span>{charCount} chars</span>
          {prompt.length > 0 && (
            <button
              type="button"
              onClick={() => setPrompt('')}
              className="flex items-center gap-1 text-[var(--text-secondary)] hover:text-rose-500 transition-colors ml-2"
              title="Clear input"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear</span>
            </button>
          )}
        </div>
      </div>

      {/* Textarea */}
      <div className="relative">
        <textarea
          id="prompt-input"
          rows={5}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Paste your prompt here... (e.g. 'Explain machine learning algorithms to high school students with relatable analogies')"
          className="w-full p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-canvas)] text-[var(--text-primary)] placeholder-[var(--text-muted)] font-mono text-sm leading-relaxed focus:outline-none focus:ring-2 focus:ring-indigo-500/30 focus:border-indigo-500 transition-all resize-y min-h-[140px]"
        />
      </div>

      {/* Example Prompts Carousel / Chips */}
      <div className="mt-3 flex items-center gap-2 overflow-x-auto pb-1 text-xs no-scrollbar">
        <span className="text-[11px] font-semibold text-[var(--text-muted)] shrink-0">Try example:</span>
        {EXAMPLE_PROMPTS.map((ex) => (
          <button
            key={ex.id}
            type="button"
            onClick={() => setPrompt(ex.prompt)}
            className="shrink-0 px-2.5 py-1 rounded-full border border-[var(--border)] bg-[var(--bg-canvas)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-indigo-400/50 transition-colors text-[11px]"
          >
            {ex.title}
          </button>
        ))}
      </div>

      {/* Controls & Submit Action */}
      <div className="mt-5 pt-4 border-t border-[var(--border)] flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Mode Selector */}
        <div className="flex flex-col sm:flex-row sm:items-center gap-3">
          <span className="text-xs font-semibold text-[var(--text-secondary)] shrink-0">Mode:</span>
          <div className="grid grid-cols-2 sm:flex items-center gap-1.5 bg-[var(--bg-canvas)] p-1 rounded-xl border border-[var(--border)]">
            {MODES.map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setMode(m.id)}
                title={m.desc}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  mode === m.id
                    ? 'bg-[var(--bg-card)] text-indigo-600 dark:text-indigo-400 shadow-sm border border-[var(--border)]'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>

          {/* Iteration Limit */}
          <div className="flex items-center gap-2 pl-2">
            <span className="text-xs font-semibold text-[var(--text-secondary)]">Max loops:</span>
            <select
              value={maxIterations}
              onChange={(e) => setMaxIterations(Number(e.target.value))}
              className="px-2 py-1 rounded-lg border border-[var(--border)] bg-[var(--bg-canvas)] text-xs font-semibold text-[var(--text-primary)] focus:outline-none focus:ring-1 focus:ring-indigo-500"
            >
              <option value={1}>1 (Fast)</option>
              <option value={2}>2 (Standard)</option>
              <option value={3}>3 (Deep, bounded)</option>
            </select>
          </div>
        </div>

        {/* Primary Action Button */}
        <button
          type="button"
          onClick={onSubmit}
          disabled={isLoading || !prompt.trim()}
          className={`inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl font-bold text-sm text-white shadow-lg shadow-indigo-500/20 transition-all ${
            isLoading || !prompt.trim()
              ? 'bg-slate-400 dark:bg-slate-700 cursor-not-allowed opacity-60 shadow-none'
              : 'bg-indigo-600 hover:bg-indigo-500 active:scale-[0.98]'
          }`}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Evaluating Agent Loop...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>Analyze & Optimize</span>
              <ArrowRight className="w-4 h-4 ml-0.5" />
            </>
          )}
        </button>
      </div>
    </div>
  )
}
