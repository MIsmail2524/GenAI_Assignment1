function exportRuns(runs) {
  const blob = new Blob([JSON.stringify(runs, null, 2)], {
    type: 'application/json',
  })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = 'restoration-inference-runs.json'
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

export default function ExperimentLog({ runs, onClear }) {
  return (
    <section className="experiment-log" aria-labelledby="run-log-title">
      <div className="experiment-log-header">
        <div>
          <h2 id="run-log-title">Inference Run Log</h2>
          <p>Successful runs saved in this browser. This is inference history, not model-training or Optuna history.</p>
        </div>
        <div className="controls">
          <button type="button" onClick={() => exportRuns(runs)} disabled={!runs.length}>
            Export JSON
          </button>
          <button type="button" onClick={onClear} disabled={!runs.length}>
            Clear log
          </button>
        </div>
      </div>

      {!runs.length ? (
        <p className="experiment-log-empty">No successful inference runs recorded yet.</p>
      ) : (
        <div className="experiment-log-table-wrap">
          <table className="experiment-log-table">
            <thead>
              <tr>
                <th>Recorded</th>
                <th>Workspace</th>
                <th>Input</th>
                <th>Run details</th>
                <th>Inference</th>
              </tr>
            </thead>
            <tbody>
              {runs.map(run => (
                <tr key={run.id}>
                  <td>{new Date(run.recordedAt).toLocaleString()}</td>
                  <td>{run.workspace}</td>
                  <td>{run.fileName}</td>
                  <td><code>{JSON.stringify(run.details)}</code></td>
                  <td>{Number(run.inferenceMs).toFixed(1)} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}