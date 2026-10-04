import { useEffect, useRef, useState } from 'react'

export default function WebcamCapture({ onCapture, onCancel }) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [ready, setReady] = useState(false)
  const [err, setErr] = useState(null)

  useEffect(() => {
    let cancelled = false
    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        if (cancelled) {
          stream.getTracks().forEach(t => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.onloadedmetadata = () => setReady(true)
        }
      } catch (e) {
        setErr(e.message || 'Could not access webcam')
      }
    }
    start()
    return () => {
      cancelled = true
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop())
      }
    }
  }, [])

  function capture() {
    const video = videoRef.current
    if (!video || !ready) return
    const canvas = document.createElement('canvas')
    canvas.width  = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0)
    canvas.toBlob(blob => {
      if (!blob) return
      const file = new File([blob], `webcam_${Date.now()}.png`, { type: 'image/png' })
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
      onCapture(file)
    }, 'image/png')
  }

  return (
    <div className="webcam-wrap">
      {err ? (
        <div className="error">{err}</div>
      ) : (
        <video ref={videoRef} autoPlay playsInline muted className="webcam-video" />
      )}
      <div className="controls" style={{ justifyContent: 'center' }}>
        <button className="primary" onClick={capture} disabled={!ready || err}>
          Capture
        </button>
        <button onClick={onCancel}>Cancel</button>
      </div>
    </div>
  )
}