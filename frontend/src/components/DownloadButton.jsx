export default function DownloadButton({ href, filename, label = 'Download result' }) {
  return (
    <a
      className="download-button"
      href={href || undefined}
      download={filename}
      aria-disabled={!href}
      onClick={event => !href && event.preventDefault()}
    >
      {label}
    </a>
  )
}