import { useState, useEffect } from 'react'
import { Header } from './components/Header'
import { PromptEditor } from './components/PromptEditor'
import { ScoreCard } from './components/ScoreCard'
import { ComparisonView } from './components/ComparisonView'
import { IterationTimeline } from './components/IterationTimeline'
import { CriticValidatorDetails } from './components/CriticValidatorDetails'
import { HistoryView } from './components/HistoryView'
import { runAgent, checkHealth, listRuns } from './services/api'
import type { AgentResponse, OptimizationMode } from './types/agent'
import { Sparkles } from 'lucide-react'

export function App() {
  const [activeTab, setActiveTab] = useState<'workspace' | 'history'>('workspace')
  const [isDark, setIsDark] = useState(() => {
    try {
      const saved = localStorage.getItem('promptlens_theme')
      return saved ? saved === 'dark' : false
    } catch {
      return false
    }
  })

  // Prompt Editor State
  const [prompt, setPrompt] = useState('')
  const [mode, setMode] = useState<OptimizationMode>('balanced')
  const [maxIterations, setMaxIterations] = useState(3)

  // Execution State
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [response, setResponse] = useState<AgentResponse | null>(null)

  // Backend Health & History Count
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null)
  const [historyCount, setHistoryCount] = useState<number>(0)

  // Sync theme
  useEffect(() => {
    try {
      localStorage.setItem('promptlens_theme', isDark ? 'dark' : 'light')
      if (isDark) {
        document.documentElement.classList.add('theme-dark')
      } else {
        document.documentElement.classList.remove('theme-dark')
      }
    } catch {
      // Ignore storage errors
    }
  }, [isDark])

  // Check backend health & count runs asynchronously
  useEffect(() => {
    let active = true
    checkHealth()
      .then(() => {
        if (active) {
          setBackendHealthy(true)
          listRuns(0, 1)
            .then((data) => {
              if (active) setHistoryCount(data.total || 0)
            })
            .catch(() => {})
        }
      })
      .catch(() => {
        if (active) setBackendHealthy(false)
      })
    return () => {
      active = false
    }
  }, [])

  // Execute Agent Loop
  const handleAnalyzeAndOptimize = async () => {
    if (!prompt.trim() || isLoading) return
    setIsLoading(true)
    setError(null)
    try {
      const result = await runAgent(prompt, mode, maxIterations)
      setResponse(result)
      // Increment history count optimistically
      setHistoryCount((prev) => prev + 1)
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : 'Failed to communicate with the PromptLens Agent Loop.'
      setError(msg)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSelectHistoryRun = (runResponse: AgentResponse) => {
    setResponse(runResponse)
    setPrompt(runResponse.original_prompt)
    setActiveTab('workspace')
  }

  return (
    <div className={`min-h-screen flex flex-col bg-[var(--bg-canvas)] text-[var(--text-primary)] ${isDark ? 'theme-dark' : ''}`}>
      {/* Top Navigation */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isDark={isDark}
        setIsDark={setIsDark}
        backendHealthy={backendHealthy}
        historyCount={historyCount}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-8">
        {activeTab === 'history' ? (
          /* History View */
          <HistoryView onSelectRun={handleSelectHistoryRun} />
        ) : (
          /* Workspace View */
          <div className="flex flex-col gap-8">
            {/* Hero / Identity Section (Shown prominently when no result, compact when result present) */}
            {!response && (
              <div className="relative overflow-hidden rounded-3xl border border-[var(--hero-border)] bg-[var(--hero-bg)] p-8 sm:p-12 text-[var(--hero-text)] shadow-xl">
                <div className="max-w-2xl relative z-10">
                  <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 mb-4">
                    <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                    <span>Autonomous Multi-Turn Agent Loop</span>
                  </div>
                  <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight leading-[1.1] mb-4">
                    See what your prompt <em className="not-italic text-indigo-400">is missing.</em>
                  </h1>
                  <p className="text-base sm:text-lg text-[var(--hero-muted)] leading-relaxed mb-6 font-normal">
                    Analyze. Improve. Test. Master your prompts with our bounded 4-stage agent pipeline: Analyzer, Optimizer, Critic, and Validator.
                  </p>
                  <div className="flex flex-wrap items-center gap-4 text-xs text-[var(--hero-muted)] font-mono">
                    <span className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-emerald-400" /> Local LLM Integration
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-indigo-400" /> SQLite Persistence
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-amber-400" /> Intent & Safety Verified
                    </span>
                  </div>
                </div>

                {/* Decorative Grid Effect */}
                <div
                  className="absolute right-0 top-0 w-1/2 h-full opacity-30 pointer-events-none hidden md:block"
                  style={{
                    backgroundImage:
                      'radial-gradient(circle, rgba(99,102,241,0.3) 1px, transparent 1px)',
                    backgroundSize: '24px 24px',
                  }}
                />
              </div>
            )}

            {/* Prompt Input & Execution */}
            <section aria-label="Prompt Input">
              <PromptEditor
                prompt={prompt}
                setPrompt={setPrompt}
                mode={mode}
                setMode={setMode}
                maxIterations={maxIterations}
                setMaxIterations={setMaxIterations}
                onSubmit={handleAnalyzeAndOptimize}
                isLoading={isLoading}
                error={error}
                setError={setError}
              />
            </section>

            {/* Live Agent Results Display */}
            {response && (
              <section aria-label="Agent Results" className="flex flex-col gap-6 animate-fadeIn">
                {/* Result Heading */}
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-[10px] font-extrabold uppercase tracking-widest text-indigo-500 block mb-1">
                      Execution Result
                    </span>
                    <h2 className="text-xl font-bold tracking-tight text-[var(--text-primary)]">
                      Optimized Prompt & Validation Report
                    </h2>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      setResponse(null)
                      window.scrollTo({ top: 0, behavior: 'smooth' })
                    }}
                    className="text-xs font-semibold text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
                  >
                    Start New Analysis →
                  </button>
                </div>

                {/* Score & Validation Badge Card */}
                <ScoreCard response={response} />

                {/* Original vs Optimized Prompt Comparison */}
                <ComparisonView response={response} />

                {/* Bounded Agent Loop Iteration Timeline */}
                {response.iterations && response.iterations.length > 0 && (
                  <IterationTimeline iterations={response.iterations} />
                )}

                {/* Critic Feedback & Validator Findings */}
                <CriticValidatorDetails response={response} />
              </section>
            )}
          </div>
        )}
      </main>

      {/* Product Footer */}
      <footer className="w-full border-t border-[var(--border)] bg-[var(--bg-card)] py-6 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-[var(--text-muted)]">
          <div className="flex items-center gap-2">
            <span className="font-bold text-[var(--text-primary)]">PromptLens AI</span>
            <span>•</span>
            <span>Production Prompt Intelligence Pipeline</span>
          </div>
          <div className="flex items-center gap-4">
            <span>Deterministic spaCy NLP</span>
            <span>•</span>
            <span>Local Quality ML</span>
            <span>•</span>
            <span>Bounded Agent Loop</span>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
