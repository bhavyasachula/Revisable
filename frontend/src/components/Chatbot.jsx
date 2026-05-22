import { useEffect, useRef, useState } from 'react'

const SUGGESTIONS = [
  'What are the main topics covered?',
  'Summarize the key concepts.',
  'What are the most important definitions?',
]

export default function Chatbot({ messages, setMessages }) {
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  const sendMessage = async text => {
    const question = (text ?? input).trim()
    if (!question || isTyping) return

    setMessages(prev => [...prev, { role: 'user', content: question }])
    setInput('')
    setIsTyping(true)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }])
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Could not reach the server. Is the backend running on port 8000?' },
      ])
    } finally {
      setIsTyping(false)
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }

  const handleKeyDown = event => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="chatbot-container">
      <div className="chat-header">
        <div className="chat-avatar">AI</div>
        <div className="chat-header-info">
          <h3>Study Assistant</h3>
          <p>Online - ready to help</p>
        </div>
      </div>

      <div className="chat-messages" id="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <span className="chat-welcome-icon">?</span>
            <h3>Ask me anything about your document</h3>
            <p>Answers stay in this chat while you move between tabs.</p>

            <div className="suggestion-list">
              {SUGGESTIONS.map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => sendMessage(suggestion)}
                  className="suggestion-chip"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              <div className="msg-avatar">
                {message.role === 'user' ? 'You' : 'AI'}
              </div>
              <div className="msg-bubble">{message.content}</div>
            </div>
          ))
        )}

        {isTyping && (
          <div className="message assistant">
            <div className="msg-avatar">AI</div>
            <div className="msg-bubble">
              <div className="typing-bubble">
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="chat-input-area">
        <textarea
          ref={inputRef}
          id="chat-input"
          className="chat-input"
          rows={1}
          placeholder="Ask a question about your PDF... (Enter to send)"
          value={input}
          onChange={event => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          id="send-message-btn"
          className="send-btn"
          onClick={() => sendMessage()}
          disabled={!input.trim() || isTyping}
          title="Send message"
        >
          Send
        </button>
      </div>
    </div>
  )
}
