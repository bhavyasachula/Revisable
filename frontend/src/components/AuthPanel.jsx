import { useState } from 'react'

export default function AuthPanel({ onAuth }) {
  const [mode, setMode] = useState('login')
  const [form, setForm] = useState({ name: '', email: '', password: '' }) // whole form thing is operating by this usestate in Usestate we have passed the object with form.name we can access the name with form.email and form.password we can access both email and password
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const isSignup = mode === 'signup'

  const updateField = event => {
    setForm(prev => ({ ...prev, [event.target.name]: event.target.value }))
  }
// submit 
  const submit = async event => {
    event.preventDefault()
    setError('')
    setLoading(true)

    try {
      const res = await fetch(`/api/${mode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      const data = await res.json()

      if (!res.ok) {
        setError(data.detail || 'Authentication failed.')
        return
      }

      onAuth(data.user)
    } catch {
      setError('Could not reach the server. Is the backend running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="auth-shell" aria-label="Account access">
      <div className="auth-panel">
        <div className="auth-copy">
          <span className="auth-kicker">Revisable Account</span>
          <h1>{isSignup ? 'Create your study space' : 'Welcome back'}</h1>
          <p>
            Save your active PDF, generated study notes, flashcards, and chat history behind your own account.
          </p>
        </div>

        <form className="auth-form" onSubmit={submit}>
          <div className="auth-tabs" role="tablist" aria-label="Authentication mode">
            <button
              type="button"
              className={mode === 'login' ? 'active' : ''}
              onClick={() => {
                setMode('login')
                setError('')
              }}
            >
              Login
            </button>
            <button
              type="button"
              className={mode === 'signup' ? 'active' : ''}
              onClick={() => {
                setMode('signup')
                setError('')
              }}
            >
              Sign up
            </button>
          </div>

          {isSignup && (
            <label className="auth-field">
              <span>Name</span>
              <input
                name="name"
                type="text"
                autoComplete="name"
                value={form.name}
                onChange={updateField}
                required
              />
            </label>
          )}

          <label className="auth-field">
            <span>Email</span>
            <input
              name="email"
              type="email"
              autoComplete="email"
              value={form.email}
              onChange={updateField}
              required
            />
          </label>

          <label className="auth-field">
            <span>Password</span>
            <input
              name="password"
              type="password"
              autoComplete={isSignup ? 'new-password' : 'current-password'}
              minLength={8}
              value={form.password}
              onChange={updateField}
              required
            />
          </label>

          {error && <p className="auth-error">{error}</p>}

          <button className="auth-submit" type="submit" disabled={loading}>
            {loading ? 'Please wait...' : isSignup ? 'Create account' : 'Login'}
          </button>
        </form>
      </div>
    </section>
  )
}
