export default function TimingCard({ ms }) {
  if (ms == null) return null
  return <div className="timing">Inference time: <b>{ms.toFixed(1)} ms</b></div>
}