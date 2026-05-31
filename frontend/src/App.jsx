import { useEffect, useState } from 'react'
import UploadZone from './components/UploadZone'
import BulletPoints from './components/BulletPoints'
import Flashcards from './components/Flashcards'
import Chatbot from './components/Chatbot'
import AuthPanel from './components/AuthPanel'

const TABS = [
  { id: 'bullets', label: 'Bullet Points', icon: 'Notes' },
  { id: 'flashcards', label: 'Flashcards', icon: 'Cards' },
  { id: 'chat', label: 'AI Chatbot', icon: 'Chat' },
]

const FEATURES = [
  {
    id: 'bullets',
    icon: 'Notes',
    iconClass: 'empty-card-icon--bullets',
    title: 'Smart Bullet Points',
    desc: 'AI extracts key topics, definitions, and concepts into organized study notes.',
  },
  {
    id: 'flashcards',
    icon: 'Cards',
    iconClass: 'empty-card-icon--flashcards',
    title: 'Flashcards',
    desc: 'Interactive flip cards for active recall - the most effective study method.',
  },
  {
    id: 'chat',
    icon: 'Chat',
    iconClass: 'empty-card-icon--chat',
    title: 'AI Chat Assistant',
    desc: 'Ask any question about your document and get instant, context-aware answers.',
  },
]

const freshStudyData = () => ({
  bullets: { data: [], loading: false, loaded: false, error: '' },
  flashcards: { data: [], loading: false, loaded: false, error: '' },
  chat: { messages: [] },
})

export default function App() {
  const [authChecked, setAuthChecked] = useState(false)
  const [user, setUser] = useState(null)
  const [uploaded, setUploaded] = useState(false)
  const [filename, setFilename] = useState('')
  const [activeTab, setActiveTab] = useState('bullets')
  const [studyData, setStudyData] = useState(freshStudyData)

  const resetStudyState = () => {
    setUploaded(false)
    setFilename('')
    setActiveTab('bullets')
    setStudyData(freshStudyData())
  }

  const loadState = async () => {
    try {
      const res = await fetch('/api/state')
      if (!res.ok) return
      const data = await res.json()

      if (data.uploaded && data.document) {
        setUploaded(true)
        setFilename(data.document.filename)
        setStudyData(prev => ({
          ...prev,
          chat: { messages: data.chat_history ?? [] },
        }))
      } else {
        resetStudyState()
      }
    } catch {
      // Upload and content actions surface connection errors when the user needs them.
    }
  }

  useEffect(() => {
    const loadSession = async () => {
      try {
        const res = await fetch('/api/me')
        const data = await res.json()

        if (res.ok && data.user) {
          setUser(data.user)
          await loadState()
        }
      } catch {
        // The auth panel shows connection problems during login/signup.
      } finally {
        setAuthChecked(true)
      }
    }

    loadSession()
  }, [])

  const handleAuth = async nextUser => {
    setUser(nextUser)
    await loadState()
  }

  const handleLogout = async () => {
    try {
      await fetch('/api/logout', { method: 'POST' })
    } finally {
      setUser(null)
      resetStudyState()
    }
  }

  function handleUploadSuccess(fname) {
    setFilename(fname)
    setUploaded(true)
    setActiveTab('bullets')
    setStudyData(freshStudyData())
  }

  const setBulletsCache = updater => {
    setStudyData(prev => ({
      ...prev,
      bullets: typeof updater === 'function' ? updater(prev.bullets) : updater,
    }))
  }

  const setFlashcardsCache = updater => {
    setStudyData(prev => ({
      ...prev,
      flashcards: typeof updater === 'function' ? updater(prev.flashcards) : updater,
    }))
  }

  const setChatMessages = updater => {
    setStudyData(prev => ({
      ...prev,
      chat: {
        messages: typeof updater === 'function'
          ? updater(prev.chat.messages)
          : updater,
      },
    }))
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">R</span>
            <span className="logo-text">Revisable</span>
          </div>
          {user ? (
            <div className="account-actions">
              <span className="account-name">{user.name}</span>
              <button className="logout-btn" type="button" onClick={handleLogout}>
                Logout
              </button>
            </div>
          ) : (
            <span className="header-badge">AI-Powered</span>
          )}
        </div>
      </header>

      <main className="main">
        {!authChecked && (
          <div className="state-box">
            <div className="spinner" />
            <p>Checking your session...</p>
          </div>
        )}

        {authChecked && !user && <AuthPanel onAuth={handleAuth} />}

        {authChecked && user && (
          <>
            {!uploaded && (
              <section className="hero">
                <div className="hero-orb hero-orb--1" />
                <div className="hero-orb hero-orb--2" />
                <div className="hero-orb hero-orb--3" />

                <div className="hero-badge">
                  <span className="hero-badge-dot" />
                  Powered by AI
                </div>

                <h1>
                  Study Smarter with{' '}
                  <span className="hero-gradient-text">AI-Generated</span>
                  <br />
                  Study Materials
                </h1>

                <p>
                  Upload any PDF and instantly get bullet-point summaries,
                  interactive flashcards, and a smart chatbot - all tailored to your document.
                </p>

                <button
                  className="hero-cta"
                  onClick={() => document.getElementById('upload-zone')?.scrollIntoView({ behavior: 'smooth' })}
                >
                  Get Started
                  <span className="hero-cta-arrow">v</span>
                </button>
              </section>
            )}

            <UploadZone onUploadSuccess={handleUploadSuccess} filename={filename} />

            {uploaded ? (
              <div className="content-section">
                <div className="tab-bar" role="tablist" aria-label="Study tools">
                  {TABS.map(tab => (
                    <button
                      key={tab.id}
                      id={`tab-${tab.id}`}
                      className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                      onClick={() => setActiveTab(tab.id)}
                      role="tab"
                      aria-selected={activeTab === tab.id}
                    >
                      <span className="tab-icon">{tab.icon}</span>
                      {tab.label}
                    </button>
                  ))}
                </div>

                <div className="tab-content">
                  {activeTab === 'bullets' && (
                    <BulletPoints cache={studyData.bullets} setCache={setBulletsCache} />
                  )}
                  {activeTab === 'flashcards' && (
                    <Flashcards cache={studyData.flashcards} setCache={setFlashcardsCache} />
                  )}
                  {activeTab === 'chat' && (
                    <Chatbot messages={studyData.chat.messages} setMessages={setChatMessages} />
                  )}
                </div>
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-cards">
                  {FEATURES.map(feat => (
                    <div key={feat.id} className="empty-card">
                      <div className={`empty-card-icon ${feat.iconClass}`}>
                        {feat.icon}
                      </div>
                      <h3>{feat.title}</h3>
                      <p>{feat.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}
