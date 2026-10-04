import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const health       = () => api.get('/health').then(r => r.data)
export const universal    = (file, corruption, severity) =>
  jsonPost('/universal-restoration', file, {
    force_corruption: corruption || 'none',
    severity,
  })
export const hardRouting  = (file, corruption, severity) =>
  jsonPost('/hard-routing', file, {
    force_corruption: corruption || 'none',
    severity,
  })
export const softMoE      = (file, corruption, severity) =>
  jsonPost('/soft-moe', file, {
    force_corruption: corruption || 'none',
    severity,
  })
export const faceToSketch = (file, style) =>
  rawPost('/face-to-sketch', file, { style })

function buildForm(file, extra = {}) {
  const fd = new FormData()
  fd.append('file', file)
  for (const [k, v] of Object.entries(extra)) fd.append(k, String(v))
  return fd
}

// endpoints that return raw PNG bytes
async function rawPost(url, file, extra = {}) {
  const fd = buildForm(file, extra)
  const res = await api.post(url, fd, { responseType: 'blob' })
  const blobUrl = URL.createObjectURL(res.data)
  const inferenceMs = parseFloat(res.headers['x-inference-ms'] || '0')
  return { blobUrl, inferenceMs }
}

// endpoints that return JSON (with base64 PNG inside)
async function jsonPost(url, file, extra = {}) {
  const fd = buildForm(file, extra)
  const { data } = await api.post(url, fd)
  return data
}