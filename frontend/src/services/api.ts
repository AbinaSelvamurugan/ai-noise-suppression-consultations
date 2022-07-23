import type { Analysis, ConfigInfo, ProcessOptions, ServerInfo, Stats } from '../types'

export const apiBase = import.meta.env.VITE_API_BASE || ''

async function parseError(response: Response): Promise<never> {
  let explanation = `Server responded ${response.status}`
  try { const json = await response.json(); explanation = typeof json.detail === 'string' ? json.detail : explanation } catch { /* HTTP error without JSON */ }
  throw new Error(explanation)
}
export async function health(): Promise<ServerInfo> {
  const r = await fetch(`${apiBase}/api/health`); if (!r.ok) return parseError(r); return r.json()
}
export async function models(): Promise<ConfigInfo> {
  const r = await fetch(`${apiBase}/api/models`); if (!r.ok) return parseError(r); return r.json()
}
export async function upload(path: string, file: File, options?: ProcessOptions): Promise<Response> {
  const fd = new FormData(); fd.append('file', file)
  if (options) fd.append('options', JSON.stringify(options))
  const r = await fetch(`${apiBase}/api/${path}`, {method: 'POST', body: fd})
  if (!r.ok) return parseError(r)
  return r
}
export async function enhance(file: File, opts: ProcessOptions): Promise<{ blob: Blob; stats: Stats }> {
  const response = await upload('enhance', file, opts)
  return {blob: await response.blob(), stats: JSON.parse(response.headers.get('X-Audio-Metadata') || '{}')}
}
export async function analyze(file: File): Promise<Analysis> {
  return (await upload('analyze', file)).json()
}
export async function evaluate(noisy: File, reference: File, opts: ProcessOptions) {
  const fd = new FormData(); fd.append('noisy', noisy); fd.append('reference', reference)
  fd.append('options', JSON.stringify(opts))
  const r = await fetch(`${apiBase}/api/evaluate`, {method: 'POST', body: fd})
  if (!r.ok) return parseError(r)
  return r.json() as Promise<{metrics: Record<string, number | null>; processing: Stats}>
}
export function download(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob); const a = document.createElement('a')
  a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 2000)
}
export function jsonDownload(obj: unknown, name: string): void {
  download(new Blob([JSON.stringify(obj, null, 2)], {type: 'application/json'}), name)
}
export function csvDownload(rows: Record<string, unknown>[], name: string): void {
  if (!rows.length) return
  const keys = Object.keys(rows[0]); const esc = (v: unknown) => `"${String(v ?? '').replaceAll('"', '""')}"`
  const csv = [keys.join(','), ...rows.map(r => keys.map(k => esc(r[k])).join(','))].join('\r\n')
  download(new Blob([csv], {type: 'text/csv'}), name)
}
