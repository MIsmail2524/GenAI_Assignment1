export default function ImageCompare({ inputUrl, outputUrl, inputLabel = 'Input', outputLabel = 'Output' }) {
  return (
    <div className="compare-grid">
      <figure>
        {inputUrl ? <img src={inputUrl} alt="input" /> : <div style={{ padding: '3rem', textAlign: 'center', background: '#f0f0f0', borderRadius: 8 }}>no input</div>}
        <figcaption>{inputLabel}</figcaption>
      </figure>
      <figure>
        {outputUrl ? <img src={outputUrl} alt="output" /> : <div style={{ padding: '3rem', textAlign: 'center', background: '#f0f0f0', borderRadius: 8 }}>no output yet</div>}
        <figcaption>{outputLabel}</figcaption>
      </figure>
    </div>
  )
}