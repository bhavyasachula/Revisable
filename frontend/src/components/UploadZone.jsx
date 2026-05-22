import { useCallback, useRef, useState } from 'react'

export default function UploadZone({ onUploadSuccess, filename }) {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const inputRef = useRef()

  const handleFile = useCallback(async file => {
    if (!file) return
    if (file.type !== 'application/pdf') {
      setError('Please upload a valid PDF file.')
      return
    }

    setError('')
    setUploading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData })
      const data = await res.json()

      if (data.filename) {
        onUploadSuccess(data.filename)
      } else {
        setError(data.detail || 'Upload failed.')
      }
    } catch {
      setError('Upload failed. Is the backend running on port 8000?')
    } finally {
      setUploading(false)
    }
  }, [onUploadSuccess])

  const onDrop = useCallback(event => {
    event.preventDefault()
    setDragging(false)
    handleFile(event.dataTransfer.files[0])
  }, [handleFile])

  const onDragOver = event => {
    event.preventDefault()
    setDragging(true)
  }

  const onDragLeave = () => setDragging(false)
  const onClick = () => {
    if (!uploading) inputRef.current?.click()
  }

  return (
    <div className="upload-section">
      <div
        id="upload-zone"
        className={`upload-zone ${dragging ? 'dragging' : ''} ${filename ? 'uploaded' : ''}`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={filename ? undefined : onClick}
      >
        <input
          ref={inputRef}
          id="pdf-file-input"
          type="file"
          accept=".pdf"
          hidden
          onChange={event => handleFile(event.target.files[0])}
        />

        {uploading ? (
          <div className="upload-loading">
            <div className="spinner" />
            <p>Processing your PDF...</p>
          </div>
        ) : filename ? (
          <div className="upload-success">
            <span className="upload-icon success">OK</span>
            <div>
              <p className="upload-filename">{filename}</p>
              <button
                type="button"
                className="upload-hint"
                onClick={event => {
                  event.stopPropagation()
                  inputRef.current?.click()
                }}
              >
                Upload a different PDF
              </button>
            </div>
          </div>
        ) : (
          <div className="upload-prompt">
            <span className="upload-icon">PDF</span>
            <h2>Drop your PDF here</h2>
            <p>or click to browse files</p>
          </div>
        )}
      </div>

      {error && <p className="upload-error">{error}</p>}
    </div>
  )
}
