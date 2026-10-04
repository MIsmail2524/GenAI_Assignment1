import { useState } from 'react'
import { universal } from '../api'
import ImageUploader from '../components/ImageUploader'
import TimingCard from '../components/TimingCard'
import DownloadButton from '../components/DownloadButton'

const CORRUPTIONS = [
  ['none',          'none (use as-is)'],
  ['clean',         'clean (no-op, baseline)'],
  ['salt_pepper',   'salt & pepper'],
  ['gaussian_blur', 'gaussian blur'],
  ['occlusion',     'occlusion'],
]

export default function UniversalRestoration({ onRunComplete }) {
  const [inputUrl, setInputUrl]   = useState(null)
  const [corruptedUrl, setCorruptedUrl] = useState(null)
  const [outputUrl, setOutputUrl] = useState(null)
  const [result, setResult]       = useState(null)
  const [corruption, setCorruption] = useState('none')
  const [severity, setSeverity]   = useState('medium')
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)

  async function run(file) {
    setLoading(true); setError(null); setCorruptedUrl(null); setOutputUrl(null); setResult(null)
    setInputUrl(URL.createObjectURL(file))
    try {
      const data = await universal(file, corruption, severity)
      setResult(data)
      setCorruptedUrl(`data:image/png;base64,${data.corrupted_png_b64}`)
      setOutputUrl(`data:image/png;base64,${data.recon_png_b64}`)
      onRunComplete?.({
        workspace: 'Universal Restoration',
        fileName: file.name,
        inferenceMs: data.inference_ms,
        details: {
          corruption: data.applied_corruption || corruption,
          severity: data.applied_severity || (corruption === 'none' ? null : severity),
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
      <h2>Universal Restoration</h2>
      <p>One autoencoder, all corruptions. Upload any image — or apply
         a corruption first to see how the universal model handles it.</p>

      <div className="controls">
        <label>
          Corruption:&nbsp;
          <select value={corruption} onChange={e => setCorruption(e.target.value)}>
            {CORRUPTIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
        </label>
        {corruption !== 'none' && corruption !== 'clean' && (
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

      <div className="compare-grid three-stage">
        <figure>
          {inputUrl ? <img src={inputUrl} alt="input" />
                    : <div className="placeholder">no input</div>}
          <figcaption>Original</figcaption>
        </figure>
        <figure>
          {corruptedUrl ? <img src={corruptedUrl} alt="corrupted input" />
                        : <div className="placeholder">no corrupted input</div>}
          <figcaption>Corrupted input</figcaption>
        </figure>
        <figure>
          {outputUrl ? <img src={outputUrl} alt="restored" />
                     : <div className="placeholder">no output yet</div>}
          <figcaption>Restored output</figcaption>
        </figure>
      </div>

      {result && (
        <div className="verdict">
          {result.clean_bypass
            ? <strong>Clean image detected; original passed through unchanged.</strong>
            : <>
                <strong>Selected corruption settings:</strong>{' '}
                {result.applied_corruption
                  ? <>{result.applied_corruption} · severity <b>{result.applied_severity}</b></>
                  : <em>none (image used as-is)</em>}
              </>}
          <TimingCard ms={result.inference_ms} />
          <DownloadButton href={outputUrl} filename="universal-restored.png" />
        </div>
      )}
    </div>
  )
}