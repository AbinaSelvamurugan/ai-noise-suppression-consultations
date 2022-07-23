import { apiBase } from '../services/api'
import type { Method } from '../types'

export interface LiveSession {
  stop(): Promise<void>
}
export async function openLive(method: Method, checkpoint: string, strength: number,
                               update: (message: string, ms?: number) => void): Promise<LiveSession> {
  const mic = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: false, channelCount: 1}})
  const context = new AudioContext()
  let ws: WebSocket | undefined
  try {
    await context.audioWorklet.addModule('/audio-worklet.js')
    const source = context.createMediaStreamSource(mic)
    const worklet = new AudioWorkletNode(context, 'consult-processor')
    source.connect(worklet); worklet.connect(context.destination)
    const target = apiBase ? new URL(apiBase, window.location.href) : new URL(window.location.href)
    const scheme = target.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = target.host
    ws = new WebSocket(`${scheme}//${host}/api/live`)
    ws.binaryType = 'arraybuffer'
    let ready = false; let seq = 0; let lost = 0
    const sent = new Map<number, number>()
    ws.onopen = () => { ws?.send(JSON.stringify({method, checkpoint, strength, sample_rate: 16000})) }
    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        try { ready = JSON.parse(event.data).status === 'ready'; if (ready) update('Live audio connected') }
        catch { update('Unexpected server handshake') }
        return
      }
      const bytes = event.data as ArrayBuffer
      if (bytes.byteLength < 8) return
      const dv = new DataView(bytes); const id = dv.getUint32(0, true)
      const serverMs = dv.getFloat32(4, true)
      const rtt = sent.has(id) ? performance.now() - sent.get(id)! : 0
      sent.delete(id)
      if (bytes.byteLength > 8) {
        const data = bytes.slice(8)
        worklet.port.postMessage({type: 'audio', samples: data}, [data])
      }
      update(`Server ${serverMs.toFixed(1)} ms · packet turnaround ${rtt.toFixed(1)} ms${lost ? ` · dropped ${lost}` : ''}`, rtt)
    }
    ws.onerror = () => update('WebSocket error; check server and microphone access')
    ws.onclose = () => { ready = false; update('Live connection closed') }
    worklet.port.onmessage = (event) => {
      if (event.data?.type === 'capture' && ready && ws?.readyState === WebSocket.OPEN) {
        if (ws.bufferedAmount > 262144) { lost++; update('Network overload: microphone packets dropped'); return }
        const data = event.data.samples as ArrayBuffer
        const packet = new ArrayBuffer(4 + data.byteLength)
        new DataView(packet).setUint32(0, seq, true)
        new Uint8Array(packet, 4).set(new Uint8Array(data))
        sent.set(seq, performance.now()); seq++
        if (sent.size > 128) sent.delete(sent.keys().next().value!)
        ws.send(packet)
      } else if (event.data?.type === 'buffer') update(`Playback queue: ${event.data.ms} ms`)
    }
    worklet.port.postMessage({type: 'live'})
    update('Opening live audio session…')
    return {stop: async () => {
      ready = false; ws?.close(1000)
      worklet.port.postMessage({type: 'stop'}); worklet.port.onmessage = null
      mic.getTracks().forEach(t => t.stop()); source.disconnect(); worklet.disconnect()
      await context.close()
    }}
  } catch (error) {
    ws?.close(); mic.getTracks().forEach(t => t.stop()); await context.close(); throw error
  }
}
