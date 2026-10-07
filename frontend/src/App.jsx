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
    <div className="container mx-auto max-w-6xl px-4 py-8">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-2 border-b border-gray-200 pb-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Multi-Corruption Restoration Suite</h1>
          <p className="subtitle text-sm text-gray-500">Tasks 1–4 · ONNX inference via FastAPI</p>
        </div>
        <span className="rounded-full bg-gray-900 px-3 py-1 font-mono text-xs text-white">ONNX Runtime · CPU</span>
      </header>

      <nav className="tabs flex flex-wrap gap-2">
        {TABS.map(([key, label]) => (
          <button key={key}
            className={`tab rounded-lg px-4 py-2 text-sm transition ${tab === key ? 'active shadow' : 'hover:border-gray-500'}`}
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