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

export default function BulletPoints({ cache, setCache }) {
  const [openCategories, setOpenCategories] = useState({})
  const bullets = cache.data

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

  useEffect(() => {
    if (!bullets.length) return

    const open = {}
    bullets.forEach((_, index) => { open[index] = true })
    setOpenCategories(open)
  }, [bullets])

  const toggle = index => {
    setOpenCategories(prev => ({ ...prev, [index]: !prev[index] }))
  }

  if (cache.loading) return <LoadingState text="Generating bullet points from your PDF..." />
  if (cache.error) return <ErrorState text={cache.error} />

  const totalPoints = bullets.reduce((acc, item) => acc + (item.points?.length ?? 0), 0)

  return (
    <div className="bullets-container">
      <div className="section-header">
        <h2>Key Topics</h2>
        <p>{totalPoints} points across {bullets.length} categories</p>
      </div>

      <div className="categories">
        {bullets.map((category, index) => (
          <div
            key={`${category.category}-${index}`}
            className="category-card"
            style={{ animationDelay: `${index * 0.05}s` }}
          >
            <button className="category-header" onClick={() => toggle(index)}>
              <div className="category-title">
                <span className="category-emoji">{category.emoji || '*'}</span>
                <h3>{category.category}</h3>
                <span className="point-count">{category.points?.length ?? 0}</span>
              </div>
              <span className={`chevron ${openCategories[index] ? 'open' : ''}`}>{'>'}</span>
            </button>

            {openCategories[index] && (
              <ul className="points-list">
                {(category.points ?? []).map((point, pointIndex) => (
                  <li
                    key={`${point}-${pointIndex}`}
                    className="point-item"
                    style={{ animationDelay: `${pointIndex * 0.03}s` }}
                  >
                    <span className="point-dot" />
                    {point}
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
