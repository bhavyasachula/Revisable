import { useEffect } from 'react'

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

export default function BulletPoints({ cache, setCache }) {
  const payload = cache.data
  const topics = Array.isArray(payload) ? payload : payload?.topics ?? []
  const overview = !Array.isArray(payload) ? payload?.overview : ''

  useEffect(() => {
    if (cache.loaded || cache.loading || cache.error) return

    const load = async () => {
      setCache(prev => ({ ...prev, loading: true, error: '' }))

      try {
        const res = await fetch('/api/bullet-points', { method: 'POST' })
        const data = await res.json()

        if (data.bullets) {
          setCache({ data: data.bullets, loading: false, loaded: true, error: '' })
        } else {
          setCache(prev => ({
            ...prev,
            loading: false,
            error: data.error || 'Failed to generate bullet points.',
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

  if (cache.loading) return <LoadingState text="Generating bullet points from your PDF..." />
  if (cache.error) return <ErrorState text={cache.error} />

  const totalPoints = topics.reduce((acc, item) => acc + (item.points?.length ?? 0), 0)
  const overviewText = overview || topics
    .map(topic => {
      const points = topic.points?.slice(0, 2).join(' ')
      return [topic.description, points].filter(Boolean).join(' ')
    })
    .filter(Boolean)
    .join(' ')

  return (
    <div className="bullets-container">
      <div className="section-header">
        <h2>Important Topics</h2>
        <p>{totalPoints} focused notes across {topics.length} topics</p>
      </div>

      <div className="topic-list">
        {topics.map((topic, index) => (
          <div
            key={`${topic.category || topic.topic}-${index}`}
            className="topic-card"
            style={{ animationDelay: `${index * 0.05}s` }}
          >
            <div className="topic-heading">
              <span className="topic-bullet">•</span>
              <div>
                <h3>{topic.category || topic.topic}</h3>
                {topic.description && <p>{topic.description}</p>}
              </div>
            </div>

            <ul className="points-list">
              {(topic.points ?? []).map((point, pointIndex) => (
                <li
                  key={`${point}-${pointIndex}`}
                  className="point-item"
                  style={{ animationDelay: `${pointIndex * 0.03}s` }}
                >
                  {point}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {overviewText && (
        <section className="overview-panel" aria-label="Overall overview">
          <h3>Description:-</h3>
          <p>{overviewText}</p>
        </section>
      )}
    </div>
  )
}
