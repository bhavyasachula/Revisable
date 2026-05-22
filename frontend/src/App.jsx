import { useEffect, useState } from 'react'
import UploadZone from './components/UploadZone'
import BulletPoints from './components/BulletPoints'
import Flashcards from './components/Flashcards'
import Chatbot from './components/Chatbot'

const TABS = [
  { id: 'bullets', label: 'Bullet Points', icon: 'Notes' },
  { id: 'flashcards', label: 'Flashcards', icon: 'Cards' },
  { id: 'chat', label: 'AI Chatbot', icon: 'Chat' },
]

const freshStudyData = () => ({
  bullets: { data: [], loading: false, loaded: false, error: '' },
  flashcards: { data: [], loading: false, loaded: false, error: '' },
  chat: { messages: [] },
})

export default function App() {
  const [uploaded, setUploaded] = useState(false)
  const [filename, setFilename] = useState('')
  const [activeTab, setActiveTab] = useState('bullets')
  const [studyData, setStudyData] = useState(freshStudyData)

  useEffect(() => {
    const loadState = async () => {
      try {
        const res = await fetch('/api/state')
        const data = await res.json()

        if (data.uploaded && data.document) {
          setUploaded(true)
          setFilename(data.document.filename)
          setStudyData(prev => ({
            ...prev,
            chat: { messages: data.chat_history ?? [] },
          }))
        }
      } catch {
        // Upload and content actions surface connection errors when the user needs them.
      }
    }

    loadState()
  }, [])

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
          <span className="header-badge">AI-Powered</span>
        </div>
      </header>

      <main className="main">
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
              {TABS.map(tab => (
                <div key={tab.id} className="empty-card">
                  <span>{tab.icon}</span>
                  <h3>{tab.label}</h3>
                  <p>
                    {tab.id === 'bullets' && 'Auto-generated key topics and summaries'}
                    {tab.id === 'flashcards' && 'Interactive flip study cards'}
                    {tab.id === 'chat' && 'Ask anything about your document'}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
