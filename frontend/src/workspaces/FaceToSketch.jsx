import { useState } from 'react'
import { faceToSketch } from '../api'
import ImageUploader from '../components/ImageUploader'
import StyleSelector from '../components/StyleSelector'
import TimingCard from '../components/TimingCard'
import WebcamCapture from '../components/WebcamCapture'
import DownloadButton from '../components/DownloadButton'

export default function FaceToSketch({ onRunComplete }) {
  const [inputUrl, setInputUrl]   = useState(null)
  const [outputUrl, setOutputUrl] = useState(null)
  const [ms, setMs]               = useState(null)
  const [style, setStyle]         = useState(0)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const [lastFile, setLastFile]   = useState(null)
  const [showWebcam, setShowWebcam] = useState(false)

  async function run(file = lastFile, selectedStyle = style) {
    if (!file) return
    setLastFile(file)
    setLoading(true); setError(null); setOutputUrl(null); setMs(null)
    setInputUrl(URL.createObjectURL(file))
    try {
      const { blobUrl, inferenceMs } = await faceToSketch(file, selectedStyle)
      setOutputUrl(blobUrl); setMs(inferenceMs)
      onRunComplete?.({
        workspace: 'Face-to-Sketch Generator',
        fileName: file.name,
        inferenceMs,
        details: { style: selectedStyle + 1 },
      })
    } catch (e) {
      setError(e.message || 'Inference failed')
    } finally {
      setLoading(false)
    }
  }

  function handleStyleChange(s) {
    setStyle(s)
    if (lastFile) run(lastFile, s)
  }

  function handleWebcamCapture(file) {
    setShowWebcam(false)
    run(file)
  }

  return (
    <div className="card">
      <h2>Face-to-Sketch Generator</h2>
      <p>Upload or capture a photograph and choose one of three sketch styles.</p>

      <div className="controls">
        <StyleSelector value={style} onChange={handleStyleChange} />
        <button className="primary" onClick={() => setShowWebcam(v => !v)}>
          {showWebcam ? 'Cancel webcam' : 'Use webcam'}
        </button>
        <DownloadButton
          href={outputUrl}
          filename={`sketch_style${style + 1}.png`}
          label="Download sketch"
        />
      </div>

      {showWebcam && (
        <WebcamCapture
          onCapture={handleWebcamCapture}
          onCancel={() => setShowWebcam(false)}
        />
      )}

      {!showWebcam && <ImageUploader onSelect={run} />}
      {loading && <p>Generating sketch…</p>}
      {error && <div className="error">{error}</div>}

      <div className="compare-grid">
        <figure>
          {inputUrl ? <img src={inputUrl} alt="photo" />
                    : <div className="placeholder">no photo</div>}
          <figcaption>Photograph</figcaption>
        </figure>
        <figure>
          {outputUrl ? <img src={outputUrl} alt="sketch" />
                     : <div className="placeholder">no sketch yet</div>}
          <figcaption>Generated sketch · Style {style + 1}</figcaption>
        </figure>
      </div>
      <TimingCard ms={ms} />
    </div>
  )
}