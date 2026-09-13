import React, { useState } from 'react'
import { Copy, Check, Download, FileText, Share2 } from 'lucide-react'
import type { AgentResponse } from '../types/agent'

interface ComparisonViewProps {
  response: AgentResponse
}

export const ComparisonView: React.FC<ComparisonViewProps> = ({ response }) => {
  const [copiedOriginal, setCopiedOriginal] = useState(false)
  const [copiedOptimized, setCopiedOptimized] = useState(false)
  const [copiedReport, setCopiedReport] = useState(false)

  const copyToClipboard = async (text: string, setFn: (val: boolean) => void) => {
    try {
      await navigator.clipboard.writeText(text)
      setFn(true)
      setTimeout(() => setFn(false), 2000)
    } catch {
      // Fallback
    }
  }

  const downloadFile = (filename: string, content: string, type: string) => {
    const blob = new Blob([content], { type })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  const exportMarkdown = () => {
    const md = `# PromptLens AI Optimization Report

**Date:** ${new Date().toISOString()}
**Quality Score:** ${response.final_score ?? 'N/A'}/100
**Validated:** ${response.is_validated ? 'YES' : 'NO'} (${response.termination_reason})
**Iterations:** ${response.iteration_count}

## Original Prompt
\`\`\`
${response.original_prompt}
\`\`\`

## Optimized Prompt
\`\`\`
${response.final_prompt}
\`\`\`

## Disclaimers
${response.disclaimers ? response.disclaimers.map((d) => `- ${d}`).join('\n') : ''}
`
    downloadFile(`promptlens-optimized-${Date.now()}.md`, md, 'text/markdown')
  }

  const exportJson = () => {
    const jsonStr = JSON.stringify(response, null, 2)
    downloadFile(`promptlens-result-${Date.now()}.json`, jsonStr, 'application/json')
  }

  const origWords = response.original_prompt.trim().split(/\s+/).length
  const optWords = response.final_prompt.trim().split(/\s+/).length

  return (
    <div className="w-full bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-5 sm:p-6 shadow-sm">
      {/* Header with Export Actions */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-4 border-b border-[var(--border)]">
        <div>
          <h2 className="text-base font-bold text-[var(--text-primary)]">
            Original vs. Optimized Prompt
          </h2>
          <p className="text-xs text-[var(--text-muted)]">
            Side-by-side prompt transformation with strict constraint and intent preservation.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={exportMarkdown}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--bg-canvas)] text-xs font-semibold text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            title="Export as Markdown"
          >
            <FileText className="w-3.5 h-3.5 text-indigo-500" />
            <span>Markdown</span>
          </button>
          <button
            type="button"
            onClick={exportJson}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--bg-canvas)] text-xs font-semibold text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            title="Download full JSON response"
          >
            <Download className="w-3.5 h-3.5 text-indigo-500" />
            <span>JSON</span>
          </button>
          <button
            type="button"
            onClick={() =>
              copyToClipboard(
                `### Optimized Prompt:\n\n${response.final_prompt}`,
                setCopiedReport
              )
            }
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--bg-canvas)] text-xs font-semibold text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            title="Copy formatted prompt"
          >
            {copiedReport ? (
              <Check className="w-3.5 h-3.5 text-emerald-500" />
            ) : (
              <Share2 className="w-3.5 h-3.5 text-indigo-500" />
            )}
            <span>{copiedReport ? 'Copied' : 'Share'}</span>
          </button>
        </div>
      </div>

      {/* Comparison Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Original Prompt Column */}
        <div className="flex flex-col border border-[var(--border)] rounded-xl bg-[var(--bg-canvas)] overflow-hidden">
          <div className="px-4 py-2.5 border-b border-[var(--border)] bg-[var(--bg-card)] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-slate-400" />
              <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-secondary)]">
                Original Prompt
              </span>
            </div>
            <div className="flex items-center gap-3 text-xs text-[var(--text-muted)]">
              <span>{origWords} words</span>
              <button
                type="button"
                onClick={() => copyToClipboard(response.original_prompt, setCopiedOriginal)}
                className="flex items-center gap-1 text-[var(--text-secondary)] hover:text-indigo-500 transition-colors"
              >
                {copiedOriginal ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copiedOriginal ? 'Copied' : 'Copy'}</span>
              </button>
            </div>
          </div>
          <div className="p-4 flex-1 text-xs font-mono whitespace-pre-wrap text-[var(--text-secondary)] leading-relaxed overflow-x-auto">
            {response.original_prompt}
          </div>
        </div>

        {/* Optimized Prompt Column */}
        <div className="flex flex-col border border-indigo-500/30 rounded-xl bg-indigo-500/[0.02] dark:bg-indigo-950/[0.08] overflow-hidden shadow-sm">
          <div className="px-4 py-2.5 border-b border-indigo-500/20 bg-[var(--bg-card)] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              <span className="text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
                Optimized Prompt
              </span>
              {response.is_validated && (
                <span className="text-[10px] bg-emerald-500/10 text-emerald-500 font-bold px-1.5 py-0.5 rounded">
                  PASS
                </span>
              )}
            </div>
            <div className="flex items-center gap-3 text-xs text-[var(--text-muted)]">
              <span>{optWords} words</span>
              <button
                type="button"
                onClick={() => copyToClipboard(response.final_prompt, setCopiedOptimized)}
                className="flex items-center gap-1 font-bold text-indigo-600 dark:text-indigo-400 hover:text-indigo-500 transition-colors"
              >
                {copiedOptimized ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copiedOptimized ? 'Copied!' : 'Copy Prompt'}</span>
              </button>
            </div>
          </div>
          <div className="p-4 flex-1 text-xs font-mono whitespace-pre-wrap text-[var(--text-primary)] leading-relaxed overflow-x-auto font-medium">
            {response.final_prompt}
          </div>
        </div>
      </div>
    </div>
  )
}
