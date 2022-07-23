/** Collect uncompressed 16-kHz mono samples from the same AudioWorklet used for live streaming. */
export type Capture = { stop: () => Promise<Float32Array>; context: AudioContext }

export function makeWav(samples: Float32Array, sr = 16000): File {
  const buffer = new ArrayBuffer(44 + samples.length * 4)
  const view = new DataView(buffer)
  const str = (offset: number, text: string) => { for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i)) }
  str(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 4, true); str(8, 'WAVE')
  str(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 3, true)
  view.setUint16(22, 1, true); view.setUint32(24, sr, true)
  view.setUint32(28, sr * 4, true); view.setUint16(32, 4, true); view.setUint16(34, 32, true)
  str(36, 'data'); view.setUint32(40, samples.length * 4, true)
  for (let i = 0; i < samples.length; i++) view.setFloat32(44 + 4 * i, samples[i], true)
  return new File([buffer], 'microphone.wav', {type: 'audio/wav'})
}

export async function startRecording(maxSeconds = 120): Promise<Capture> {
  const tracks = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: false, channelCount: 1}})
  const context = new AudioContext()
  try {
    await context.audioWorklet.addModule('/audio-worklet.js')
    const source = context.createMediaStreamSource(tracks)
    const worklet = new AudioWorkletNode(context, 'consult-processor')
    const silent = context.createGain(); silent.gain.value = 0
    source.connect(worklet); worklet.connect(silent); silent.connect(context.destination)
    const received: Float32Array[] = []; let length = 0
    worklet.port.onmessage = event => {
      if (event.data?.type === 'capture') {
        const chunk = new Float32Array(event.data.samples)
        if (length + chunk.length <= maxSeconds * 16000) { received.push(chunk); length += chunk.length }
      }
    }
    return {context, stop: async () => {
      const stopped = new Promise<void>(resolve => {
        const previous = worklet.port.onmessage
        worklet.port.onmessage = event => {
          if (event.data?.type === 'stopped') resolve()
          else if (previous) previous.call(worklet.port, event)
        }
      })
      worklet.port.postMessage({type: 'stop'})
      await Promise.race([stopped, new Promise<void>(resolve => setTimeout(resolve, 500))])
      tracks.getTracks().forEach(t => t.stop())
      source.disconnect(); worklet.disconnect(); silent.disconnect(); worklet.port.onmessage = null
      await context.close()
      const combined = new Float32Array(length); let offset = 0
      for (const chunk of received) { combined.set(chunk, offset); offset += chunk.length }
      return combined
    }}
  } catch (error) { tracks.getTracks().forEach(t => t.stop()); await context.close(); throw error }
}
