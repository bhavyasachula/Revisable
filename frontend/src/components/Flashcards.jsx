import { useEffect, useState } from 'react'

function LoadingState({ text }) {
  return (
    <div className="state-box">
      <div className="spinner" />
      <p>{text}</p>
    </div>
  )
}

function ErrorState({ text }) {
  return (
    <div className="state-box error-state">
      <span className="err-icon">!</span>
      <p>{text}</p>
    </div>
  )
}

export default function Flashcards({ cache, setCache }) {
  const [currentIdx, setCurrentIdx] = useState(0)
  const [flipped, setFlipped] = useState(false)
  const [visited, setVisited] = useState(new Set([0]))
  const cards = cache.data

  useEffect(() => {
    if (cache.loaded || cache.loading || cache.error) return

    const load = async () => {
      setCache(prev => ({ ...prev, loading: true, error: '' }))

      try {
        const res = await fetch('/api/flashcards', { method: 'POST' })
        const data = await res.json()

        if (data.flashcards) {
          setCache({ data: data.flashcards, loading: false, loaded: true, error: '' })
        } else {
          setCache(prev => ({
            ...prev,
            loading: false,
            error: data.error || 'Failed to generate flashcards.',
          }))
        }
      } catch {
        setCache(prev => ({
          ...prev,
          loading: false,
          error: 'Could not reach the server. Is the backend running?',
        }))
      }
    }

    load()
  }, [cache.error, cache.loaded, cache.loading, setCache])

  const goTo = index => {
    setCurrentIdx(index)
    setFlipped(false)
    setVisited(prev => new Set([...prev, index]))
  }

  const prev = () => currentIdx > 0 && goTo(currentIdx - 1)
  const next = () => currentIdx < cards.length - 1 && goTo(currentIdx + 1)

  const handleFlip = () => {
    setVisited(prev => new Set([...prev, currentIdx]))
    setFlipped(value => !value)
  }

  useEffect(() => {
    const handler = event => {
      if (event.key === 'ArrowRight') next()
      if (event.key === 'ArrowLeft') prev()
      if (event.key === ' ') {
        event.preventDefault()
        handleFlip()
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [currentIdx, cards.length])

  if (cache.loading) return <LoadingState text="Generating flashcards from your PDF..." />
  if (cache.error) return <ErrorState text={cache.error} />
  if (!cards.length) return <ErrorState text="No flashcards generated." />

  const card = cards[currentIdx]
  const remaining = cards.length - visited.size

  return (
    <div className="flashcards-container">
      <div className="flashcards-header section-header">
        <div>
          <h2>Flashcards</h2>
          <p>Click the card to flip. Use arrow keys to move.</p>
        </div>
        <div className="flashcard-meta">
          <span className="fc-progress">
            <strong>{currentIdx + 1}</strong> / {cards.length}
          </span>
          {card.difficulty && (
            <span className={`badge ${card.difficulty?.toLowerCase() ?? 'medium'}`}>
              {card.difficulty}
            </span>
          )}
        </div>
      </div>

      <div
        id="flashcard"
        className="flip-card-wrap"
        onClick={handleFlip}
        role="button"
        aria-label={flipped ? 'Show question' : 'Show answer'}
        tabIndex={0}
      >
        <div className={`flip-card-inner ${flipped ? 'flipped' : ''}`}>
          <div className="flip-card-front">
            <span className="card-label">Question</span>
            <p className="card-question">{card.question}</p>
            <p className="card-hint">Click to reveal answer</p>
          </div>
          <div className="flip-card-back">
            <span className="card-back-label">Answer</span>
            <p className="card-answer">{card.answer}</p>
          </div>
        </div>
      </div>

      <div className="card-nav">
        <button
          id="fc-prev-btn"
          className="nav-btn"
          onClick={prev}
          disabled={currentIdx === 0}
          aria-label="Previous card"
        >
          {'<'}
        </button>

        <div className="progress-dots">
          {cards.map((_, index) => (
            <button
              key={index}
              className={[
                'progress-dot',
                index === currentIdx ? 'active' : '',
                visited.has(index) && index !== currentIdx ? 'visited' : '',
              ].join(' ')}
              onClick={() => goTo(index)}
              title={`Card ${index + 1}`}
              aria-label={`Go to card ${index + 1}`}
            />
          ))}
        </div>

        <button
          id="fc-next-btn"
          className="nav-btn"
          onClick={next}
          disabled={currentIdx === cards.length - 1}
          aria-label="Next card"
        >
          {'>'}
        </button>
      </div>

      <p className="fc-tip">
        {remaining === 0
          ? "You've reviewed all cards. Great work!"
          : `${remaining} card${remaining !== 1 ? 's' : ''} left to review`}
      </p>
    </div>
  )
}
