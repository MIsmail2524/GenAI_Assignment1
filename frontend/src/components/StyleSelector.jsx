export default function StyleSelector({ value, onChange }) {
  return (
    <select value={value} onChange={e => onChange(Number(e.target.value))}>
      <option value={0}>Style 1</option>
      <option value={1}>Style 2</option>
      <option value={2}>Style 3</option>
    </select>
  )
}