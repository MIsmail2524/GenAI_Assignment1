export default function RoutingBar({ weights, highlight }) {
  const entries = Object.entries(weights)
  const max = Math.max(...entries.map(([, v]) => v), 1e-6)
  return (
    <div style={{ marginTop: '1rem' }}>
      {entries.map(([label, value]) => {
        const isHighlighted = label === highlight
        return (
          <div key={label} className="routing-bar">
            <span className="label">{label}</span>
            <span
              className={`bar ${isHighlighted ? 'bar-dominant' : ''}`}
              style={{ width: `${(value / max) * 300}px` }}
            />
            <span className="value">{(value * 100).toFixed(1)}%</span>
          </div>
        )
      })}
    </div>
  )
}