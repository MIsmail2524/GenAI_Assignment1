export default function ImageUploader({ onSelect, label = 'Click or drop an image' }) {
  return (
    <label className="uploader">
      <input type="file" accept="image/*" hidden
        onChange={e => e.target.files[0] && onSelect(e.target.files[0])} />
      {label}
    </label>
  )
}