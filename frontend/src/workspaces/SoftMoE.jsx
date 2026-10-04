import { useState } from 'react'
import { softMoE } from '../api'
import ImageUploader from '../components/ImageUploader'
import RoutingBar from '../components/RoutingBar'
import TimingCard from '../components/TimingCard'
import DownloadButton from '../components/DownloadButton'

const BRANCH_LABEL = {
  clean:          'identity (clean)',
  salt_pepper:    'salt-and-pepper expert',
  gaussian_blur:  'gaussian-blur expert',
  occlusion:      'occlusion expert',
}

const CORRUPTIONS = [
  ['none', 'use uploaded image'],
  ['clean', 'clean baseline'],
  ['salt_pepper', 'salt-and-pepper'],
  ['gaussian_blur', 'Gaussian blur'],
  ['occlusion', 'occlusion'],
]

export default function SoftMoE({ onRunComplete }) {
  const [inputUrl, setInputUrl] = useState(null)
  const [result, setResult]     = useState(null)
  const [corruption, setCorruption] = useState('none')
  const [severity, setSeverity] = useState('medium')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)

  async function run(file) {
    setLoading(true); setError(null); setResult(null)
    setInputUrl(URL.createObjectURL(file))
    try {
      const data = await softMoE(file, corruption, severity)
      setResult(data)
      onRunComplete?.({
        workspace: 'Soft Mixture-of-Experts Restoration',
        fileName: file.name,
        inferenceMs: data.inference_ms,
        details: {
          appliedCorruption: data.applied_corruption || corruption,
          severity: data.applied_severity || (corruption === 'none' ? null : severity),
          dominantExpert: data.dominant_expert,
          routingWeights: data.routing_weights,
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
      <h2>Soft Mixture-of-Experts Restoration</h2>
      <p>All branches contribute, weighted by a differentiable gate.</p>

      <div className="controls">
        <label>
          Apply corruption first:{' '}
          <select value={corruption} onChange={e => setCorruption(e.target.value)}>
            {CORRUPTIONS.map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </label>
        {corruption !== 'none' && corruption !== 'clean' && (
          <label>
            Severity:{' '}
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
              <img src={`data:image/png;base64,${result.recon_png_b64}`}
                   alt="restored" />
              <figcaption>Restored output</figcaption>
            </figure>
          </div>

          <div className="verdict">
            <div className="verdict-row">
              <span className="verdict-key">Dominant expert</span>
              <span className="verdict-val">
                <b>{BRANCH_LABEL[result.dominant_expert]}</b>{' '}
                ({(result.routing_weights[result.dominant_expert] * 100).toFixed(1)}%)
              </span>
            </div>
            {result.applied_corruption && (
              <div className="verdict-row">
                <span className="verdict-key">Applied corruption</span>
                <span className="verdict-val">
                  {result.applied_corruption} · {result.applied_severity}
                </span>
              </div>
            )}
            {result.clean_bypass && (
              <div className="verdict-row">
                <span className="verdict-key">Restoration</span>
                <span className="verdict-val">Clean input passed through unchanged</span>
              </div>
            )}
            <TimingCard ms={result.inference_ms} />
            <DownloadButton
              href={`data:image/png;base64,${result.recon_png_b64}`}
              filename="soft-moe-restored.png"
            />
          </div>

          <h3 style={{ marginTop: '1.5rem' }}>Routing weights</h3>
          <RoutingBar
            weights={result.routing_weights}
            highlight={result.dominant_expert}
          />

          <h4 style={{ marginTop: '1rem' }}>Contribution breakdown</h4>
          <ul className="breakdown">
            {Object.entries(result.routing_weights)
              .sort((a, b) => b[1] - a[1])
              .map(([k, v]) => (
                <li key={k}>
                  <span>{BRANCH_LABEL[k]}</span>
                  <span className="pct">{(v * 100).toFixed(1)}%</span>
                </li>
              ))}
          </ul>
        </>
      )}
    </div>
  )
}