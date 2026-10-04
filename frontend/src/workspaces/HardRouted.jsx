import { useState } from 'react'
import { hardRouting } from '../api'
import ImageUploader from '../components/ImageUploader'
import RoutingBar from '../components/RoutingBar'
import TimingCard from '../components/TimingCard'
import DownloadButton from '../components/DownloadButton'

const CORRUPTIONS = [
  ['none',          'none (use as-is)'],
  ['salt_pepper',   'salt & pepper'],
  ['gaussian_blur', 'gaussian blur'],
  ['occlusion',     'occlusion'],
]

const EXPERT_LABEL = {
  identity:       'identity (bypass — image left untouched)',
  salt_pepper:    'salt-and-pepper specialist',
  gaussian_blur:  'gaussian-blur specialist',
  occlusion:      'occlusion specialist',
}

export default function HardRouted({ onRunComplete }) {
  const [inputUrl, setInputUrl]     = useState(null)
  const [result, setResult]         = useState(null)
  const [corruption, setCorruption] = useState('none')
  const [severity, setSeverity]     = useState('medium')
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)

  async function run(file) {
    setLoading(true); setError(null); setResult(null)
    setInputUrl(URL.createObjectURL(file))
    try {
      const data = await hardRouting(file, corruption, severity)
      setResult(data)
      onRunComplete?.({
        workspace: 'Hard-Routed Restoration',
        fileName: file.name,
        inferenceMs: data.inference_ms,
        details: {
          appliedCorruption: data.applied_corruption || corruption,
          severity: data.applied_severity || (corruption === 'none' ? null : severity),
          prediction: data.predicted,
          expert: data.expert_used,
          probabilities: data.probabilities,
          cleanBypass: data.clean_bypass,
        },
      })
    } catch (e) {
      setError(e.message || 'Inference failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <h2>Hard-Routed Restoration</h2>
      <p>Classifier picks one specialist. Clean inputs bypass all experts.</p>

      <div className="controls">
        <label>
          Apply corruption first:&nbsp;
          <select value={corruption} onChange={e => setCorruption(e.target.value)}>
            {CORRUPTIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
        </label>
        {corruption !== 'none' && (
          <label>
            Severity:&nbsp;
            <select value={severity} onChange={e => setSeverity(e.target.value)}>
              <option value="low">low</option>
              <option value="medium">medium</option>
              <option value="high">high</option>
            </select>
          </label>
        )}
      </div>

      <ImageUploader onSelect={run} />
      {loading && <p>Running inference…</p>}
      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <div className="compare-grid three-stage" style={{ marginTop: '1rem' }}>
            <figure>
              <img src={inputUrl} alt="input" />
              <figcaption>Original</figcaption>
            </figure>
            <figure>
              <img
                src={`data:image/png;base64,${result.corrupted_png_b64}`}
                alt="corrupted input"
              />
              <figcaption>
                Corrupted input
                {result.applied_corruption &&
                  <> · {result.applied_corruption} ({result.applied_severity})</>}
              </figcaption>
            </figure>
            <figure>
              <img src={`data:image/png;base64,${result.recon_png_b64}`} alt="restored" />
              <figcaption>Restored output</figcaption>
            </figure>
          </div>

          <div className="verdict">
            <div className="verdict-row">
              <span className="verdict-key">Predicted corruption</span>
              <span className="verdict-val"><b>{result.predicted}</b></span>
            </div>
            <div className="verdict-row">
              <span className="verdict-key">Expert selected</span>
              <span className="verdict-val">{EXPERT_LABEL[result.expert_used]}</span>
            </div>
            <TimingCard ms={result.inference_ms} />
            <DownloadButton
              href={`data:image/png;base64,${result.recon_png_b64}`}
              filename="hard-routed-restored.png"
            />
          </div>

          <h3 style={{ marginTop: '1.5rem' }}>Classifier probabilities</h3>
          <RoutingBar weights={result.probabilities} />
        </>
      )}
    </div>
  )
}