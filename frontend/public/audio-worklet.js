/* Worklet: microphone downmix/decimation and bounded monitor playback, no UI-thread DSP. */
class ConsultProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.frames = []
    this.recordBuffer = new Float32Array(1024)
    this.recordLength = 0
    this.phase = 0
    this.downSum = 0
    this.downCount = 0
    this.playQueue = new Float32Array(16384)
    this.queueHead = 0
    this.queueLength = 0
    this.playGain = 0
    this.playCursor = 0
    this.playPrev = 0
    this.playNext = 0
    this.playStarted = false
    this.live = false
    this.active = true
    this.lastReport = 0
    this.port.onmessage = (event) => {
      const data = event.data
      if (data?.type === 'audio') {
        const arr = new Float32Array(data.samples)
        for (let i = 0; i < arr.length; i++) {
          if (this.queueLength === this.playQueue.length) {
            this.queueHead = (this.queueHead + 1) % this.playQueue.length
            this.queueLength--
          }
          this.playQueue[(this.queueHead + this.queueLength) % this.playQueue.length] = arr[i]
          this.queueLength++
        }
      } else if (data?.type === 'live') this.live = true
      else if (data?.type === 'stop') {
        if (this.recordLength) {
          const tail = this.recordBuffer.slice(0, this.recordLength)
          this.port.postMessage({type: 'capture', samples: tail.buffer}, [tail.buffer])
          this.recordLength = 0
        }
        this.active = false; this.queueLength = 0
        this.port.postMessage({type: 'stopped'})
      }
    }
  }
  next() {
    if (!this.queueLength) return 0
    const value = this.playQueue[this.queueHead]
    this.queueHead = (this.queueHead + 1) % this.playQueue.length
    this.queueLength--
    return value
  }
  process(inputs, outputs) {
    if (!this.active) return false
    const input = inputs[0]?.[0]
    if (input) {
      const r = 16000 / sampleRate
      for (let i = 0; i < input.length; i++) {
        this.downSum += input[i]; this.downCount++; this.phase += r
        if (this.phase >= 1) {
          this.phase -= 1
          this.recordBuffer[this.recordLength++] = this.downSum / this.downCount
          this.downSum = 0; this.downCount = 0
          if (this.recordLength === 1024) {
            const result = this.recordBuffer.slice()
            this.port.postMessage({type: 'capture', samples: result.buffer}, [result.buffer])
            this.recordLength = 0
          }
        }
      }
    }
    const output = outputs[0]?.[0]
    if (output && this.live) {
      if (!this.playStarted && this.queueLength > 2200) {
        this.playStarted = true; this.playPrev = this.next(); this.playNext = this.next()
      }
      if (this.playStarted) {
        const step = 16000 / sampleRate
        for (let i = 0; i < output.length; i++) {
          this.playGain += ((this.queueLength ? 1 : 0) - this.playGain) * 0.004
          this.playGain = Math.max(0, Math.min(1, this.playGain))
          output[i] = (this.playPrev + (this.playNext - this.playPrev) * this.playCursor) * this.playGain
          this.playCursor += step
          if (this.playCursor >= 1) {
            this.playCursor -= 1; this.playPrev = this.playNext; this.playNext = this.next()
          }
        }
        if (this.queueLength === 0) this.playStarted = false
      }
      this.lastReport++
      if (this.lastReport >= 40) {
        this.port.postMessage({type: 'buffer', ms: Math.round(this.queueLength / 16)})
        this.lastReport = 0
      }
    }
    return true
  }
}
registerProcessor('consult-processor', ConsultProcessor)
