import { useEffect, useRef, useState } from 'react'
import { Activity, AudioLines, BarChart3, ChevronRight, CircleHelp, Download, FileAudio, Gauge,
  Headphones, LayoutDashboard, Mic, Pause, Play, Radio, RefreshCw, Settings as SettingsIcon, ShieldCheck,
  SlidersHorizontal, UploadCloud, Waves, XCircle } from 'lucide-react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { unzipSync, strFromU8 } from 'fflate'
import type { Analysis, ConfigInfo, Page, ProcessOptions, ServerInfo, Stats } from './types'
import { analyze, csvDownload, download, enhance, evaluate, health, jsonDownload, models, upload } from './services/api'
import { makeWav, startRecording, type Capture } from './audio/record'
import { differenceWav } from './audio/residual'
import { openLive, type LiveSession } from './audio/live'
import { Spectrogram, Waveform } from './components/Visuals'
import './style.css'

const defaultOptions: ProcessOptions = { method: 'spectral', checkpoint: '', strength: 1.4, n_fft: 512,
  hop: 128, minimum_gain: .08, noise_seconds: .35, noise_start_seconds: 0,
  smooth_time: 1, smooth_frequency: 1 }
const navigation: {page: Page; name: string; icon: typeof Activity}[] = [
  {page: 'dashboard', name: 'Overview', icon: LayoutDashboard}, {page: 'enhance', name: 'Enhancement studio', icon: Waves},
  {page: 'compare', name: 'Model comparison', icon: AudioLines}, {page: 'live', name: 'Live microphone', icon: Radio},
  {page: 'evaluate', name: 'Evaluation', icon: BarChart3}, {page: 'settings', name: 'System settings', icon: SettingsIcon},
]
function errorMessage(error: unknown): string { return error instanceof Error ? error.message : 'Unexpected processing error' }
function timeLabel(t: number) { return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}` }
function KeyValue({label, value}: {label: string; value: string | number}) {
  return <div className="key-value"><span>{label}</span><strong>{value}</strong></div>
}
function MethodControls({options, setOptions, info}: {options: ProcessOptions; setOptions: (next: ProcessOptions) => void; info: ConfigInfo | null}) {
  const change = (fields: Partial<ProcessOptions>) => setOptions({...options, ...fields})
  return <section className="panel"><h3><SlidersHorizontal size={17}/> Enhancement controls</h3>
    <label className="field-label">Algorithm</label>
    <div className="segmented"><button aria-pressed={options.method === 'spectral'} className={options.method === 'spectral' ? 'selected' : ''} onClick={() => change({method: 'spectral'})}>Spectral gate</button>
      <button disabled={!info?.models.length} aria-pressed={options.method === 'neural'} className={options.method === 'neural' ? 'selected' : ''} onClick={() => change({method: 'neural'})}>PyTorch neural</button></div>
    {options.method === 'spectral' ? <>
      <label className="field-label">Reduction profile</label>
      <div className="segmented"><button onClick={() => change({strength: .6, minimum_gain: .25, smooth_time: 2, smooth_frequency: 2})}>Gentle</button><button onClick={() => change({strength: 1.4, minimum_gain: .08, smooth_time: 1, smooth_frequency: 1})}>Balanced</button><button onClick={() => change({strength: 2.5, minimum_gain: .05, smooth_time: 1.5, smooth_frequency: 1.5})}>Strong</button></div>
      <label className="field-label" htmlFor="strength">Reduction strength <b>{options.strength.toFixed(1)}</b></label>
      <input id="strength" type="range" min="0" max="4" step=".1" value={options.strength} onChange={e => change({strength: +e.target.value})}/>
      <p className="hint">Higher strength suppresses more noise but can distort quiet speech.</p>
      <label className="field-label" htmlFor="fft">FFT size</label><select id="fft" value={options.n_fft} onChange={e => change({n_fft: +e.target.value, hop: +e.target.value / 4})}>
        {[256, 512, 1024, 2048].map(n => <option key={n} value={n}>{n} samples</option>)}</select>
      <label className="field-label" htmlFor="calibration">Assumed noise-only duration (s)</label>
      <input id="calibration" type="number" step=".05" min=".05" max="10" value={options.noise_seconds} onChange={e => change({noise_seconds: +e.target.value})}/>
      <label className="field-label" htmlFor="calibration-start">Noise-only segment begins (s)</label>
      <input id="calibration-start" type="number" step=".05" min="0" value={options.noise_start_seconds} onChange={e => change({noise_start_seconds: +e.target.value})}/>
      <label className="field-label" htmlFor="gain">Minimum gain <b>{options.minimum_gain.toFixed(2)}</b></label>
      <input id="gain" type="range" min="0" max="1" step=".01" value={options.minimum_gain} onChange={e => change({minimum_gain: +e.target.value})}/>
      <label className="field-label" htmlFor="smooth">Mask smoothing <b>{options.smooth_time.toFixed(1)}</b></label>
      <input id="smooth" type="range" min="0" max="8" step=".25" value={options.smooth_time} onChange={e => change({smooth_time: +e.target.value, smooth_frequency: +e.target.value})}/>
      <p className="hint">Choose a noise-only segment if possible; speech during calibration may be removed.</p>
    </> : <>
      <label className="field-label" htmlFor="checkpoint">Trained checkpoint</label>
      <select id="checkpoint" value={options.checkpoint} onChange={e => change({checkpoint: e.target.value})}>
        <option value="">Default trained checkpoint</option>
        {info?.models.map(m => <option key={m.name} value={m.name}>{m.name} · {m.parameters.toLocaleString()} parameters</option>)}
      </select>
      <p className="hint">Uses the locally installed trained Conv–GRU checkpoint; random weights are never substituted.</p>
    </>}
  </section>
}
function AudioPlayer({original, enhanced, residual, labels = ['Original', 'Enhanced']}: {original: string; enhanced: string; residual?: string; labels?: [string, string]}) {
  const player = useRef<HTMLAudioElement>(null)
  const [choice, setChoice] = useState<'original' | 'enhanced' | 'residual'>('original')
  const [playing, setPlaying] = useState(false)
  const [position, setPosition] = useState(0)
  const [duration, setDuration] = useState(0)
  const [blind, setBlind] = useState(false)
  const [blindMap, setBlindMap] = useState<{A: 'original' | 'enhanced'; B: 'original' | 'enhanced'}>({A: 'original', B: 'enhanced'})
  const [reveal, setReveal] = useState(false)
  function choose(next: 'original' | 'enhanced' | 'residual') {
    const t = player.current?.currentTime || 0; const wasPlaying = !player.current?.paused
    setChoice(next)
    requestAnimationFrame(() => { if (player.current) { player.current.currentTime = Math.min(t, player.current.duration || t); if (wasPlaying) void player.current.play().catch(() => setPlaying(false)) } })
  }
  useEffect(() => { setChoice('original'); setPlaying(false); setPosition(0) }, [original])
  function startBlind() {
    const bit = crypto.getRandomValues(new Uint8Array(1))[0] % 2
    setBlindMap(bit ? {A: 'enhanced', B: 'original'} : {A: 'original', B: 'enhanced'})
    setBlind(true); setReveal(false)
  }
  const src = choice === 'original' ? original : choice === 'enhanced' ? enhanced : residual || original
  return <section className="panel playback-panel"><div className="section-heading"><h3><Headphones size={18}/> Listening comparison</h3>
    {enhanced && <button className="text-button" onClick={() => blind ? (setBlind(false), setReveal(false)) : startBlind()}>{blind ? 'Exit blind A/B' : 'Start blind A/B'}</button>}</div>
    <audio key={src} ref={player} src={src} onTimeUpdate={e => setPosition(e.currentTarget.currentTime)} onDurationChange={e => setDuration(e.currentTarget.duration || 0)} onEnded={() => setPlaying(false)} />
    <div className="playbar"><button className="play-button" aria-label={playing ? 'Pause' : 'Play'} onClick={() => { if (!player.current) return; if (player.current.paused) { void player.current.play().then(() => setPlaying(true)).catch(() => setPlaying(false)) } else {player.current.pause(); setPlaying(false)} }}>{playing ? <Pause size={18}/> : <Play size={18}/>}</button>
      <span className="time">{timeLabel(position)}</span><input aria-label="Seek playback" type="range" min="0" max={duration || 1} step=".01" value={position} onChange={e => {if (player.current) player.current.currentTime = +e.target.value; setPosition(+e.target.value)}} />
      <span className="time">{timeLabel(duration)}</span></div>
    <div className="listen-switch">
      {blind ? <><button className={choice === blindMap.A ? 'active' : ''} onClick={() => choose(blindMap.A)}>Sample A</button><button className={choice === blindMap.B ? 'active' : ''} onClick={() => choose(blindMap.B)}>Sample B</button>
        <button className="text-button" onClick={() => setReveal(!reveal)}>{reveal ? `${blindMap.A === 'original' ? 'A: original · B: enhanced' : 'A: enhanced · B: original'}` : 'Reveal identity'}</button></> : <>
        <button className={choice === 'original' ? 'active' : ''} onClick={() => choose('original')}>{labels[0]}</button>
        <button disabled={!enhanced} className={choice === 'enhanced' ? 'active' : ''} onClick={() => choose('enhanced')}>{labels[1]}</button>
        <button disabled={!residual} className={choice === 'residual' ? 'active' : ''} onClick={() => choose('residual')}>Residual (difference)</button>
      </>}
    </div><p className="hint">Switching sources preserves playback position (within media duration). The residual is a signal difference, not isolated background noise.</p>
  </section>
}
function StatusCard({info, status}: {info: ConfigInfo | null; status: ServerInfo | null}) {
  return <div className="status-grid">
    <div className="stat"><span>BACKEND STATUS</span><strong className={status ? 'green' : 'red'}>{status ? 'Connected' : 'Offline'}</strong><small>{status ? 'API responding' : 'Start FastAPI first'}</small></div>
    <div className="stat"><span>NEURAL MODEL</span><strong>{info?.models.length ? 'Ready' : 'Not installed'}</strong><small>{info?.models.length || 0} compatible checkpoint(s)</small></div>
    <div className="stat"><span>PROCESSING DEVICE</span><strong>{status?.device.toUpperCase() || 'Unknown'}</strong><small>{status?.sample_rate || 16000} Hz target</small></div>
    <div className="stat"><span>APPLICATION</span><strong>v{status?.version || '0.1.0'}</strong><small>Consultation Audio Studio</small></div>
  </div>
}

export default function App() {
  const [page, setPage] = useState<Page>('dashboard')
  const [status, setStatus] = useState<ServerInfo | null>(null)
  const [info, setInfo] = useState<ConfigInfo | null>(null)
  const [options, setOptions] = useState<ProcessOptions>(defaultOptions)
  const [file, setFile] = useState<File | null>(null)
  const [sourceUrl, setSourceUrl] = useState('')
  const [outputUrl, setOutputUrl] = useState('')
  const [outputBlob, setOutputBlob] = useState<Blob | null>(null)
  const [residualUrl, setResidualUrl] = useState('')
  const [sourceAnalysis, setSourceAnalysis] = useState<Analysis>()
  const [outputAnalysis, setOutputAnalysis] = useState<Analysis>()
  const [stats, setStats] = useState<Stats>()
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [recording, setRecording] = useState(false)
  const recordRef = useRef<Capture | null>(null)
  const [liveStatus, setLiveStatus] = useState('Microphone not active')
  const [liveTurnaround, setLiveTurnaround] = useState<number | null>(null)
  const [liveRunning, setLiveRunning] = useState(false)
  const liveRef = useRef<LiveSession | null>(null)
  const [compareResults, setCompareResults] = useState<Record<string, Stats | {available: false; reason: string}> | null>(null)
  const [compWavs, setCompWavs] = useState<Record<string, string>>({})
  const [compareZip, setCompareZip] = useState<Blob | null>(null)
  const [reference, setReference] = useState<File | null>(null)
  const [evaluation, setEvaluation] = useState<{metrics: Record<string, number | null>; processing: Stats} | null>(null)
  const [history, setHistory] = useState<Record<string, number>[]>([])
  const fileInput = useRef<HTMLInputElement>(null)

  async function refreshStatus() {
    try { const [s, m] = await Promise.all([health(), models()]); setStatus(s); setInfo(m) }
    catch { setStatus(null); setInfo(null) }
  }
  useEffect(() => { void refreshStatus(); const interval = setInterval(() => {void refreshStatus()}, 15000); return () => clearInterval(interval) }, [])
  useEffect(() => () => { if (sourceUrl) URL.revokeObjectURL(sourceUrl); if (outputUrl) URL.revokeObjectURL(outputUrl); if (residualUrl) URL.revokeObjectURL(residualUrl) }, [sourceUrl, outputUrl, residualUrl])
  useEffect(() => () => {void liveRef.current?.stop(); void recordRef.current?.stop()}, [])
  useEffect(() => () => Object.values(compWavs).forEach(url => URL.revokeObjectURL(url)), [compWavs])

  async function selectFile(item: File) {
    setError('')
    if (!/\.(wav|flac|ogg)$/i.test(item.name)) { setError('Choose a WAV, FLAC or OGG file'); return }
    if (item.size > (info?.max_upload_mb || 20) * 1024 * 1024) {setError('File exceeds server upload limit'); return}
    setFile(item); setOutputUrl(''); setResidualUrl(''); setOutputBlob(null); setOutputAnalysis(undefined); setStats(undefined)
    setCompareResults(null); setCompWavs({}); setCompareZip(null); setEvaluation(null)
    setSourceUrl(URL.createObjectURL(item)); setSourceAnalysis(undefined); setBusy('Validating and analyzing audio…')
    try {setSourceAnalysis(await analyze(item))} catch (e) {setError(errorMessage(e)); setFile(null)} finally {setBusy('')}
  }
  async function process() {
    if (!file) return
    setBusy('Running audio enhancement…'); setError('')
    try {
      const {blob, stats: metadata} = await enhance(file, options)
      const resultFile = new File([blob], 'enhanced.wav', {type: 'audio/wav'})
      const analysis = await analyze(resultFile)
      setOutputBlob(blob); setOutputUrl(URL.createObjectURL(blob)); setOutputAnalysis(analysis); setStats(metadata)
      try { setResidualUrl(URL.createObjectURL(await differenceWav(file, blob))) } catch { setResidualUrl('') }
    } catch (e) {setError(errorMessage(e))} finally {setBusy('')}
  }
  async function toggleRecording() {
    setError('')
    if (recording && recordRef.current) {
      try {const samples = await recordRef.current.stop(); if (samples.length) await selectFile(makeWav(samples))}
      catch (e) {setError(errorMessage(e))} finally {setRecording(false); recordRef.current = null}
    } else {
      try {recordRef.current = await startRecording(info?.max_duration_s || 120); setRecording(true)}
      catch (e) {setError(`Microphone unavailable: ${errorMessage(e)}`)}
    }
  }
  async function compareMethods() {
    if (!file) return
    setBusy('Processing identical audio with available methods…'); setError('')
    try {
      const response = await upload('compare', file, options)
      const archiveBytes = await response.arrayBuffer()
      const archive = unzipSync(new Uint8Array(archiveBytes))
      const results = JSON.parse(strFromU8(archive['measurements.json']))
      setCompareResults(results.results)
      const urls: Record<string, string> = {}
      for (const name of ['original', 'spectral', 'neural']) {
        if (archive[`${name}.wav`]) {
          const bytes = archive[`${name}.wav`]
          const copy = new Uint8Array(bytes.byteLength); copy.set(bytes)
          urls[name] = URL.createObjectURL(new Blob([copy.buffer], {type: 'audio/wav'}))
        }
      }
      setCompWavs(urls)
      setCompareZip(new Blob([archiveBytes], {type: 'application/zip'}))
    } catch (e) {setError(errorMessage(e))} finally {setBusy('')}
  }
  async function runEvaluation() {
    if (!file || !reference) return
    setBusy('Computing paired-reference metrics…'); setError('')
    try {setEvaluation(await evaluate(file, reference, options))}
    catch (e) {setError(errorMessage(e))} finally {setBusy('')}
  }
  async function toggleLive() {
    if (liveRunning) {
      await liveRef.current?.stop(); liveRef.current = null; setLiveRunning(false); setLiveStatus('Microphone stopped'); return
    }
    setError(''); setLiveStatus('Requesting microphone permission…')
    try {liveRef.current = await openLive(options.method, options.checkpoint, options.strength,
      (message, ms) => {setLiveStatus(message); if (ms !== undefined) setLiveTurnaround(ms)})
      setLiveRunning(true)
    } catch (e) {setError(errorMessage(e)); setLiveStatus('Unable to start live processing')}
  }
  function fileDrop(e: React.DragEvent) {e.preventDefault(); if (e.dataTransfer.files[0]) void selectFile(e.dataTransfer.files[0])}
  const fileInfo = sourceAnalysis ? `${sourceAnalysis.duration_s.toFixed(2)} s · ${sourceAnalysis.sample_rate.toLocaleString()} Hz processing` : 'WAV · FLAC · OGG'
  const title = navigation.find(n => n.page === page)?.name || ''
  return <div className="app-shell">
    <aside className="sidebar"><div className="brand"><div className="brand-icon"><AudioLines size={24}/></div><div><b>ClearVoice<span> AI</span></b><small>CONSULTATION AUDIO</small></div></div>
      <div className="nav-title">WORKSPACE</div><nav aria-label="Main navigation">{navigation.map(item => <button key={item.page} className={page === item.page ? 'nav-item active' : 'nav-item'} onClick={() => setPage(item.page)}><item.icon size={18}/>{item.name}{page === item.page && <ChevronRight size={14} className="nav-chevron"/>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="privacy"><ShieldCheck size={20}/><div><b>Privacy first</b><p>Recordings are processed in memory and are not stored by default.</p></div></div><span>v0.1.0 · Research software</span></div>
    </aside>
    <main className="main"><header className="topbar"><div className="breadcrumb">WORKSPACE <ChevronRight size={13}/> <b>{title.toUpperCase()}</b></div><div className="top-actions"><span className={status ? 'connection connected' : 'connection'}><span className="status-dot"/>{status ? 'Backend online' : 'Backend offline'}</span><button className="icon-button" aria-label="Refresh server status" onClick={() => void refreshStatus()}><RefreshCw size={17}/></button></div></header>
      <div className="content">
      <div className="page-heading"><div><span className="eyebrow">AI AUDIO SIGNAL PROCESSING STUDIO</span><h1>{page === 'dashboard' ? 'Welcome to ClearVoice' : title}</h1><p>{page === 'dashboard' ? 'Cleaner conversations, supported by measurable audio engineering.' : 'Signal processing with transparent measurements.'}</p></div><span className="build-pill">PYTORCH + FASTAPI</span></div>
      {error && <div className="alert error" role="alert"><XCircle size={18}/><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}
      {busy && <div className="alert activity" role="status"><span className="spinner"/> {busy}</div>}
      {page === 'dashboard' && <>
        <StatusCard info={info} status={status}/>
        <div className="hero panel"><div><span className="eyebrow">START YOUR SESSION</span><h2>Hear the difference.<br/><em>Keep the voice.</em></h2><p>Upload or record a consultation-style sample, remove background noise, and inspect waveform and spectral changes.</p>
          <button className="primary" onClick={() => setPage('enhance')}>Open enhancement studio <ChevronRight size={17}/></button></div><div className="hero-graphic"><div className="sound-orbit"><AudioLines size={66}/></div></div></div>
        <div className="feature-grid">{[{name: 'Enhancement workspace', text: 'Process WAV, FLAC and OGG files with STFT gating or a trained PyTorch model.', page: 'enhance' as Page, icon: Waves},
          {name: 'Compare algorithms', text: 'Apply both algorithms to the same input, export timing measurements.', page: 'compare' as Page, icon: Gauge},
          {name: 'Live microphone', text: 'Stream 16 kHz samples through a stateful overlap/add WebSocket pipeline.', page: 'live' as Page, icon: Mic},
          {name: 'Reference evaluation', text: 'Compute SI-SDR, spectral distance and optional STOI against aligned clean speech.', page: 'evaluate' as Page, icon: BarChart3}].map(c => <button className="feature panel" key={c.page} onClick={() => setPage(c.page)}><c.icon size={23}/><b>{c.name}</b><p>{c.text}</p><span>Explore <ChevronRight size={13}/></span></button>)}</div>
        <div className="notice"><CircleHelp size={18}/> Noise suppression may distort or remove quiet words. Enhanced audio is not a verbatim medical or counselling record and is not a diagnostic tool.</div>
      </>}
      {(page === 'enhance' || page === 'compare') && <div className="workspace">
        <div className="workspace-main"><section className="panel"><div className="section-heading"><h3><FileAudio size={18}/> Source recording</h3>{file && <span className="tag">{fileInfo}</span>}</div>
          <input className="sr-only" type="file" ref={fileInput} accept=".wav,.flac,.ogg,audio/wav,audio/flac,audio/ogg" aria-label="Choose audio file" onChange={e => {if (e.target.files?.[0]) void selectFile(e.target.files[0])}}/>
          <div className="dropzone" role="button" tabIndex={0} onKeyDown={e => {if (e.key === 'Enter') fileInput.current?.click()}} onDragOver={e => e.preventDefault()} onDrop={fileDrop} onClick={() => fileInput.current?.click()}><UploadCloud size={31}/>
            <b>{file ? file.name : 'Drop your audio recording here'}</b><p>{file ? 'Click to replace recording' : 'Or click to browse from your computer'}</p><small>Mono/stereo · WAV / FLAC / OGG · Max {info?.max_upload_mb || 20} MB</small></div>
          <div className="upload-actions"><button className="secondary" onClick={() => fileInput.current?.click()}><UploadCloud size={16}/> Choose file</button><button className={recording ? 'danger' : 'secondary'} onClick={() => void toggleRecording()}><Mic size={16}/>{recording ? 'Stop recording' : 'Record microphone'}</button></div>
          {recording && <p className="record-indicator"><span className="status-dot"/> Microphone recording · browser permission active</p>}
        </section>
        <section className="panel"><div className="section-heading"><h3><AudioLines size={18}/> Signal inspection</h3><span className="tag">16 kHz target · mono</span></div>
          <div className="signal-grid"><Waveform data={sourceAnalysis} title="Original waveform"/><Waveform data={outputAnalysis} title="Enhanced waveform" accent="#4cdbc1"/>
            <Spectrogram data={sourceAnalysis} title="Original spectrogram"/><Spectrogram data={outputAnalysis} title="Enhanced spectrogram"/></div>
        </section>
        {sourceUrl && <AudioPlayer original={sourceUrl} enhanced={outputUrl} residual={residualUrl}/>}
        {stats && <section className="panel"><h3><Gauge size={18}/> Processing measurements</h3><div className="mini-stats"><KeyValue label="Processing time" value={`${stats.processing_ms.toFixed(1)} ms`}/><KeyValue label="Real-time factor" value={stats.real_time_factor.toFixed(3)}/><KeyValue label="Output length" value={`${stats.duration_s.toFixed(2)} s`}/><KeyValue label="Output format" value={stats.format}/></div>
          <p className="hint">Wall-clock measurement of local inference, excluding upload and browser playback. RTF = processing seconds / audio seconds.</p></section>}
        {page === 'compare' && <section className="panel"><h3><BarChart3 size={18}/> Same-input algorithm comparison</h3><p className="hint">Runs spectral and trained neural methods on the identical input and exports measured durations. Clean-reference quality metrics require the Evaluation page.</p>
          <button className="secondary" disabled={!file || !!busy} onClick={() => void compareMethods()}>Compare both algorithms</button>
          {compareResults && <><div className="upload-actions"><button className="secondary" disabled={!compareZip} onClick={() => compareZip && download(compareZip, 'comparison.zip')}><Download size={16}/> Download comparison ZIP</button></div><div className="comparison-grid">{Object.entries(compareResults).map(([method, data]) => <div className="compare-card" key={method}><h4>{method.toUpperCase()}</h4>{'available' in data && !data.available ? <p>Not available: {data.reason}</p> : <><KeyValue label="Inference" value={`${(data as Stats).processing_ms.toFixed(1)} ms`}/><KeyValue label="RTF" value={(data as Stats).real_time_factor.toFixed(3)}/><KeyValue label="Sample rate" value={(data as Stats).sample_rate}/></>}
                  {compWavs[method] && <audio controls preload="metadata" src={compWavs[method]} aria-label={`${method} playback`}/>}</div>)}</div>{compWavs.spectral && <AudioPlayer original={compWavs.spectral} enhanced={compWavs.neural || compWavs.original} labels={compWavs.neural ? ['Spectral gate', 'Trained neural'] : ['Spectral gate', 'Original']}/>}<p className="hint">Same-input timing and the WAVs are available in the comparison ZIP.</p></>}
        </section>}
        </div><aside className="controls-side"><MethodControls options={options} setOptions={setOptions} info={info}/>
          <section className="panel action-panel"><h3>Process recording</h3><p>Generate a downloadable 16 kHz mono float32 WAV using the selected algorithm.</p>
            <button className="primary full" disabled={!file || !!busy || (options.method === 'neural' && !info?.models.length)} onClick={() => void process()}><AudioLines size={18}/> Enhance audio</button>
            <button className="secondary full" disabled={!outputBlob} onClick={() => outputBlob && download(outputBlob, 'clearvoice-enhanced.wav')}><Download size={17}/> Download WAV</button>
          </section><div className="notice"><ShieldCheck size={17}/> Audio files are not retained by the server. Listening to the output is essential to check speech integrity.</div></aside>
      </div>}
      {page === 'live' && <div className="live-grid"><section className="panel live-panel"><div className="live-illustration"><div className={liveRunning ? 'mic-ring active' : 'mic-ring'}><Mic size={44}/></div></div>
        <h2>{liveRunning ? 'Microphone is streaming' : 'Live noise suppression'}</h2><p>Capture audio through an AudioWorklet, process in server-side 16 kHz STFT frames, and monitor the result with headphones.</p>
        <button className={liveRunning ? 'danger' : 'primary'} onClick={() => void toggleLive()}>{liveRunning ? 'Stop live processing' : 'Allow microphone & start live processing'}</button>
        <div className="live-status" role="status"><span className={liveRunning ? 'status-dot' : ''}/> {liveStatus}</div>
        {liveTurnaround !== null && <div className="live-latency">Last observed packet turnaround: <strong>{liveTurnaround.toFixed(1)} ms</strong></div>}
        <p className="hint">Turnaround includes WebSocket/server processing, not microphone hardware and output speaker delays. Use headphones to prevent feedback.</p>
      </section><div><MethodControls options={options} setOptions={setOptions} info={info}/><div className="notice">Browser audio capture varies by device. If packets overflow, they may be dropped; stop and restart after network instability. No persistent live audio storage.</div></div></div>}
      {page === 'evaluate' && <div className="evaluation-grid"><section className="panel"><h3><BarChart3 size={18}/> Clean-reference evaluation</h3><p>Provide sample-aligned noisy and clean recordings. Reference-dependent metrics require a reference.</p>
        <label className="field-label" htmlFor="eval-noisy">Noisy test recording</label><input id="eval-noisy" type="file" accept=".wav,.flac,.ogg" onChange={e => e.target.files?.[0] && void selectFile(e.target.files[0])}/>
        <label className="field-label" htmlFor="eval-clean">Matching clean reference</label><input id="eval-clean" type="file" accept=".wav,.flac,.ogg" onChange={e => setReference(e.target.files?.[0] || null)}/>
        <button className="primary" disabled={!file || !reference || !!busy} onClick={() => void runEvaluation()}>Evaluate enhancement</button>
        {evaluation && <><div className="metric-table">{Object.entries(evaluation.metrics).map(([key, val]) => <div className="key-value" key={key}><span>{key.replaceAll('_',' ')}</span><strong>{val == null ? 'Not available' : val.toFixed(3)}</strong></div>)}</div>
          <div className="upload-actions"><button className="secondary" onClick={() => jsonDownload(evaluation, 'evaluation.json')}>Export JSON</button><button className="secondary" onClick={() => csvDownload([{...evaluation.metrics, ...evaluation.processing}], 'evaluation.csv')}>Export CSV</button></div></>}
        <div className="notice"><CircleHelp size={18}/> SI-SDR depends on alignment; STOI only when the optional dependency and valid durations are available. PESQ requires a separately licensed evaluator.</div>
      </section><section className="panel"><h3><Activity size={18}/> Training / validation loss history</h3><p>Import the <code>history.csv</code> generated by the PyTorch trainer.</p>
        <input aria-label="Import training history CSV" type="file" accept=".csv" onChange={async e => {const csv = await e.target.files?.[0]?.text(); if (!csv) return
          const lines = csv.trim().split(/\r?\n/), fields = lines[0].split(',')
          setHistory(lines.slice(1).map(line => Object.fromEntries(line.split(',').map((v, i) => [fields[i], +v]))))
        }}/>
        {history.length ? <div className="chart-area"><ResponsiveContainer width="100%" height={265}><LineChart data={history}><CartesianGrid stroke="#304052"/><XAxis dataKey="epoch" stroke="#98a5b8"/><YAxis stroke="#98a5b8"/><Tooltip/><Line dataKey="train_total" stroke="#7b8bfd" dot={false}/><Line dataKey="val_total" stroke="#45d4b7" dot={false}/></LineChart></ResponsiveContainer></div> : <div className="chart-empty">No training history loaded.</div>}
        {history.length > 0 && <button className="secondary" onClick={() => csvDownload(history, 'training-history.csv')}>Export plotted history</button>}
      </section></div>}
      {page === 'settings' && <div className="settings-grid"><section className="panel"><h3><SettingsIcon size={18}/> Backend configuration</h3><button className="secondary" onClick={() => void refreshStatus()}><RefreshCw size={16}/> Refresh status</button>
        <div className="metric-table"><KeyValue label="API health" value={status?.status || 'Unavailable'}/><KeyValue label="Version" value={status?.version || 'Unknown'}/>
          <KeyValue label="Processing device" value={info?.device || 'Unavailable'}/><KeyValue label="Target rate" value={`${info?.configured_sample_rate || '—'} Hz`}/>
          <KeyValue label="Allowed inputs" value={info?.supported_formats.join(', ') || 'Unavailable'}/><KeyValue label="Max upload" value={`${info?.max_upload_mb || '—'} MB`}/>
          <KeyValue label="Max duration" value={`${info?.max_duration_s || '—'} s`}/><KeyValue label="Max concurrency" value={info?.max_concurrent || '—'}/>
        </div></section><section className="panel"><h3><AudioLines size={18}/> Installed trained models</h3>{info?.models.length ? info.models.map(m => <div className="model-record" key={m.name}><b>{m.name}</b><p>{m.architecture} · {m.parameters.toLocaleString()} parameters · {m.sample_rate} Hz</p><pre>{JSON.stringify(m.config, null, 2)}</pre></div>) : <div className="chart-empty">No trained neural checkpoint installed. Spectral gating remains available.</div>}
        {Boolean(info?.rejected_checkpoints) && <p className="hint">{info?.rejected_checkpoints} incompatible local checkpoint(s) rejected.</p>}
        <p className="hint">To install a trained checkpoint, follow README training steps and restart the API. Never upload checkpoints through the public API.</p></section></div>}
      <footer>ClearVoice AI · Audio enhancement research project · No clinical certification or verbatim-transcription guarantee.</footer>
      </div>
    </main>
  </div>
}
