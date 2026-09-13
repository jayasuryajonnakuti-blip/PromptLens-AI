import React from 'react'
import { Sparkles, Clock, Sun, Moon, CheckCircle2, XCircle } from 'lucide-react'

interface HeaderProps {
  activeTab: 'workspace' | 'history'
  setActiveTab: (tab: 'workspace' | 'history') => void
  isDark: boolean
  setIsDark: (dark: boolean) => void
  backendHealthy: boolean | null
  historyCount?: number
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  isDark,
  setIsDark,
  backendHealthy,
  historyCount,
}) => {
  return (
    <header className="w-full border-b border-[var(--border)] bg-[var(--bg-card)] sticky top-0 z-20 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center text-white font-bold shadow-md shadow-indigo-500/20 text-lg">
            P
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-base tracking-tight text-[var(--text-primary)]">
                PromptLens
              </span>
              <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-500">
                AI Agent
              </span>
            </div>
            <p className="text-[11px] text-[var(--text-muted)] hidden sm:block">
              Analyze. Improve. Test. Master your prompts.
            </p>
          </div>
        </div>

        {/* Center Navigation */}
        <nav className="flex items-center gap-1 bg-[var(--bg-canvas)] p-1 rounded-lg border border-[var(--border)]">
          <button
            type="button"
            onClick={() => setActiveTab('workspace')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-semibold transition-all ${
              activeTab === 'workspace'
                ? 'bg-[var(--bg-card)] text-[var(--text-primary)] shadow-sm'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
            Workspace
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('history')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-semibold transition-all ${
              activeTab === 'history'
                ? 'bg-[var(--bg-card)] text-[var(--text-primary)] shadow-sm'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            <Clock className="w-3.5 h-3.5 text-indigo-500" />
            History
            {typeof historyCount === 'number' && historyCount > 0 && (
              <span className="ml-1 px-1.5 py-0.2 rounded-full text-[10px] bg-indigo-500/10 text-indigo-500 font-bold">
                {historyCount}
              </span>
            )}
          </button>
        </nav>

        {/* Right Actions */}
        <div className="flex items-center gap-3">
          {/* Backend Status Pill */}
          <div
            className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium border border-[var(--border)] bg-[var(--bg-canvas)] text-[var(--text-secondary)]"
            title={backendHealthy ? 'FastAPI pipeline ready' : 'Backend offline or unreachable'}
          >
            {backendHealthy === true ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                <span>Agent Ready</span>
              </>
            ) : backendHealthy === false ? (
              <>
                <XCircle className="w-3.5 h-3.5 text-rose-500" />
                <span>Backend Offline</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                <span>Connecting...</span>
              </>
            )}
          </div>

          {/* Theme Toggle */}
          <button
            type="button"
            onClick={() => setIsDark(!isDark)}
            aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            className="w-9 h-9 rounded-lg border border-[var(--border)] bg-[var(--bg-card)] flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
          >
            {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-600" />}
          </button>
        </div>
      </div>
    </header>
  )
}
