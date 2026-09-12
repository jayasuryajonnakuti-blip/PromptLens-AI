import { useState } from 'react'
import './App.css'

const navigation = [
  { label: 'Dashboard', icon: 'grid' },
  { label: 'Prompt Analyzer', icon: 'scan' },
  { label: 'Optimizer', icon: 'wand' },
  { label: 'Playground', icon: 'terminal' },
  { label: 'History', icon: 'clock' },
  { label: 'Saved Prompts', icon: 'bookmark' },
  { label: 'Templates', icon: 'layers' },
  { label: 'Analytics', icon: 'chart' },
  { label: 'Settings', icon: 'settings' },
]

function Icon({ name }: { name: string }) {
  const paths: Record<string, string> = {
    grid: 'M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z',
    scan: 'M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3M8 12h8',
    wand: 'm15 4 5 5M13 6l5 5M4 20l2.5-6.5L16 4l4 4-9.5 9.5L4 20Z',
    terminal: 'm7 8 4 4-4 4M13 16h4',
    clock: 'M12 7v5l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z',
    bookmark: 'M6 4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18l-6-4-6 4V4Z',
    layers: 'm12 3 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 16l9 5 9-5',
    chart: 'M4 19V5M4 19h16M8 16v-3M12 16V8M16 16v-6',
    settings: 'M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7ZM19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-1.8 1.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-2.6V20a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1-1.8-1.8.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.6-1H7.2v-2.6h.2a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1 1.8-1.8.1.1a1.7 1.7 0 0 0 1.9.3 1.7 1.7 0 0 0 1-1.6V5h2.6v.2a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1 1.8 1.8-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v2.6H21a1.7 1.7 0 0 0-1.6 1Z',
    sun: 'M12 3V1M12 23v-2M4.2 4.2 2.8 2.8M21.2 21.2l-1.4-1.4M3 12H1M23 12h-2M4.2 19.8l-1.4 1.4M21.2 2.8l-1.4 1.4M17 12a5 5 0 1 1-10 0 5 5 0 0 1 10 0Z',
    moon: 'M20.5 15.5A8.5 8.5 0 0 1 8.5 3.5 8.5 8.5 0 1 0 20.5 15.5Z',
    menu: 'M4 7h16M4 12h16M4 17h16',
    arrow: 'M5 12h14M13 6l6 6-6 6',
  }

  return (
    <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d={paths[name]} />
    </svg>
  )
}

function App() {
  const [isDark, setIsDark] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)

  return (
    <div className={`app-shell ${isDark ? 'theme-dark' : ''}`}>
      <aside className={`sidebar ${isSidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">P</div>
          <div>
            <span className="brand-name">PromptLens</span>
            <span className="brand-type">AI workspace</span>
          </div>
        </div>

        <nav className="primary-nav" aria-label="Primary navigation">
          <span className="nav-label">Workspace</span>
          {navigation.slice(0, 4).map((item) => (
            <button
              className={`nav-item ${item.label === 'Dashboard' ? 'active' : ''}`}
              key={item.label}
              type="button"
              onClick={() => setIsSidebarOpen(false)}
            >
              <Icon name={item.icon} />
              <span>{item.label}</span>
            </button>
          ))}
          <span className="nav-label nav-label-secondary">Library</span>
          {navigation.slice(4, 8).map((item) => (
            <button className="nav-item" key={item.label} type="button" onClick={() => setIsSidebarOpen(false)}>
              <Icon name={item.icon} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <button className="nav-item" type="button" onClick={() => setIsSidebarOpen(false)}>
            <Icon name="settings" />
            <span>Settings</span>
          </button>
          <div className="workspace-card">
            <span className="workspace-status"></span>
            <div>
              <span className="workspace-title">Personal workspace</span>
              <span className="workspace-caption">Local foundation</span>
            </div>
            <span className="workspace-menu">•••</span>
          </div>
        </div>
      </aside>

      {isSidebarOpen && <button className="sidebar-backdrop" type="button" aria-label="Close navigation" onClick={() => setIsSidebarOpen(false)} />}

      <div className="main-shell">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-menu" type="button" aria-label="Open navigation" onClick={() => setIsSidebarOpen(true)}>
              <Icon name="menu" />
            </button>
            <div className="breadcrumb"><span>Workspace</span><span className="breadcrumb-divider">/</span><strong>Dashboard</strong></div>
          </div>
          <div className="topbar-actions">
            <span className="status-indicator"><span></span>All systems normal</span>
            <button className="icon-button" type="button" aria-label={`Switch to ${isDark ? 'light' : 'dark'} theme`} onClick={() => setIsDark(!isDark)}>
              <Icon name={isDark ? 'sun' : 'moon'} />
            </button>
            <button className="avatar" type="button" aria-label="Open profile menu">JD</button>
          </div>
        </header>

        <main className="content-area">
          <div className="content-heading">
            <div>
              <span className="eyebrow">Good morning, Jordan</span>
              <h1>Build prompts with clarity.</h1>
              <p className="heading-copy">A focused workspace for turning rough ideas into reliable instructions.</p>
            </div>
            <button className="button button-secondary" type="button"><span>⌘</span> Quick search</button>
          </div>

          <section className="hero-panel" aria-labelledby="hero-title">
            <div className="hero-copy">
              <div className="hero-kicker"><span className="kicker-dot"></span>Prompt intelligence, without the noise</div>
              <h2 id="hero-title">See what your prompt<br /><em>is missing.</em></h2>
              <p>Build prompts that work.</p>
              <button className="button button-primary" type="button">Analyze Prompt <Icon name="arrow" /></button>
            </div>
            <div className="hero-grid" aria-hidden="true">
              <div className="grid-orbit orbit-one"></div>
              <div className="grid-orbit orbit-two"></div>
              <div className="signal-card signal-card-top"><span className="signal-line"></span><span>Intent</span><strong>Clear</strong></div>
              <div className="signal-card signal-card-bottom"><span className="signal-line signal-line-muted"></span><span>Structure</span><strong>Ready</strong></div>
              <div className="hero-spark">✦</div>
            </div>
          </section>

          <section className="empty-section" aria-labelledby="recent-title">
            <div className="section-heading"><div><span className="eyebrow">Your workspace</span><h2 id="recent-title">Recent prompts</h2></div><button className="text-button" type="button">View history <Icon name="arrow" /></button></div>
            <div className="empty-card">
              <div className="empty-icon"><Icon name="scan" /></div>
              <h3>Your prompt history will appear here</h3>
              <p>Start with an analysis to build a clearer, more capable prompt.</p>
              <button className="button button-tertiary" type="button">Create your first prompt <Icon name="arrow" /></button>
            </div>
          </section>
        </main>
        <footer className="app-footer"><span>PromptLens AI</span><span>Foundation workspace <b></b> v0.1</span></footer>
        </div>
    </div>
  )
}

export default App
