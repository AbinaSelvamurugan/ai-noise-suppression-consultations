import { makeWav } from './record'

/** Difference signal in the same output sample grid, not an isolated noise track. */
export async function differenceWav(original: File, enhanced: Blob): Promise<File> {
  const sampleRate = 16000
  const decode = new AudioContext()
  try {
    const referenceBuffer = await decode.decodeAudioData(await original.arrayBuffer())
    const enhancedBuffer = await decode.decodeAudioData(await enhanced.arrayBuffer())
    const len = enhancedBuffer.length
    const ctx = new OfflineAudioContext(1, len, sampleRate)
    const source = ctx.createBufferSource()
    source.buffer = referenceBuffer
    source.connect(ctx.destination)
    source.start(0)
    const aligned = (await ctx.startRendering()).getChannelData(0)
    const target = enhancedBuffer.getChannelData(0)
    const residual = new Float32Array(len)
    for (let i = 0; i < len; i++) residual[i] = aligned[i] - target[i]
    return makeWav(residual, sampleRate)
  } finally { await decode.close() }
}
