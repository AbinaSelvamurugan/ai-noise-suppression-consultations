import { useEffect, useRef } from 'react'
import type { Analysis } from '../types'

export function Waveform({data, accent = '#7b8bfd', title}: {data?: Analysis; accent?: string; title: string}) {
  const values = data?.peak || []
  return <div className="visual"><div className="visual-title">{title}<span>Waveform · peak envelope</span></div>
    <svg viewBox="0 0 600 130" preserveAspectRatio="none" role="img" aria-label={`${title} waveform`}>
      <line x1="0" x2="600" y1="65" y2="65" stroke="#354357" strokeWidth="1" />
      {values.map((v, i) => <line key={i} x1={i * 600 / values.length} x2={i * 600 / values.length} y1={65 - Math.min(1, v) * 60} y2={65 + Math.min(1, v) * 60} stroke={accent} strokeWidth={Math.max(1, 600 / values.length * .85)} />)}
    </svg>
    <div className="visual-axis"><span>0 s</span><span>{data ? `${data.duration_s.toFixed(2)} s` : 'No audio'}</span></div>
  </div>
}
export function Spectrogram({data, title}: {data?: Analysis; title: string}) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = ref.current; const matrix = data?.spectrogram_db
    if (!canvas || !matrix?.length) return
    const ctx = canvas.getContext('2d'); if (!ctx) return
    const width = matrix[0].length; const height = matrix.length
    const image = ctx.createImageData(width, height)
    for (let f = 0; f < height; f++) for (let t = 0; t < width; t++) {
      const x = Math.max(0, Math.min(1, (matrix[f][t] + 95) / 95))
      const index = ((height - f - 1) * width + t) * 4
      image.data[index] = Math.round(17 + 210 * Math.pow(x, 1.6))
      image.data[index + 1] = Math.round(30 + 153 * x)
      image.data[index + 2] = Math.round(49 + 202 * Math.sqrt(x))
      image.data[index + 3] = 255
    }
    const temp = document.createElement('canvas'); temp.width = width; temp.height = height
    temp.getContext('2d')?.putImageData(image, 0, 0)
    ctx.imageSmoothingEnabled = false
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(temp, 0, 0, canvas.width, canvas.height)
  }, [data])
  return <div className="visual"><div className="visual-title">{title}<span>Log magnitude · dB</span></div>
    <div className="spectrogram-container"><div className="freq-axis"><span>8k</span><span>4k</span><span>0</span></div><canvas ref={ref} width={600} height={160} role="img" aria-label={`${title} log magnitude spectrogram`} /></div>
    <div className="visual-axis"><span>0 s</span><span>{data ? `${data.duration_s.toFixed(2)} s` : 'No audio'}</span></div>
  </div>
}
