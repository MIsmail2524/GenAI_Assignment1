import { useEffect, useState } from 'react'
import UniversalRestoration from "./workspaces/UniversalRestoration";
import HardRouted from "./workspaces/HardRouted";
import SoftMoE from "./workspaces/SoftMoE";
import FaceToSketch from "./workspaces/FaceToSketch";
import ExperimentLog from './components/ExperimentLog'

const RUNS_STORAGE_KEY = 'restoration.inferenceRuns'

function readRuns() {
  try {
    const stored = JSON.parse(localStorage.getItem(RUNS_STORAGE_KEY) || '[]')
    return Array.isArray(stored) ? stored : []
  } catch {
    return []
  }
}
const TABS = [
  ['universal', 'Universal Restoration',               UniversalRestoration],
  ['hard',      'Hard-Routed Restoration',             HardRouted],
  ['soft',      'Soft Mixture-of-Experts Restoration', SoftMoE],
  ['sketch',    'Face-to-Sketch Generator',            FaceToSketch],
]

export default function App() {
  const [tab, setTab] = useState('universal')
  const [runs, setRuns] = useState(readRuns)
  const [, , View] = TABS.find(t => t[0] === tab)

  useEffect(() => {
    localStorage.setItem(RUNS_STORAGE_KEY, JSON.stringify(runs))
  }, [runs])

  function recordRun(run) {
    setRuns(current => [{
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      recordedAt: new Date().toISOString(),
      ...run,
    }, ...current].slice(0, 100))
  }

  return (
    <div className="container">
      <h1>Multi-Corruption Restoration Suite</h1>
      <p className="subtitle">Tasks 1–4 · ONNX inference via FastAPI</p>

      <nav className="tabs">
        {TABS.map(([key, label]) => (
          <button key={key}
            className={`tab ${tab === key ? 'active' : ''}`}
            onClick={() => setTab(key)}>
            {label}
          </button>
        ))}
      </nav>

      <View onRunComplete={recordRun} />
      <ExperimentLog runs={runs} onClear={() => setRuns([])} />
    </div>
  )
}