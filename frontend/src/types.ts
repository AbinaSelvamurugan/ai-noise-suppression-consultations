export type Method = 'spectral' | 'neural'
export type Page = 'dashboard' | 'enhance' | 'compare' | 'live' | 'evaluate' | 'settings'
export interface Stats { method: Method; model: string; sample_rate: number; device: string;
  processing_ms: number; real_time_factor: number; duration_s: number; channels: number;
  format: string; source?: {sample_rate: number; channels: number; frames: number; duration: number; peak: number} }
export interface ModelInfo { name: string; architecture: string; parameters: number; sample_rate: number; config: Record<string, number> }
export interface ServerInfo { status: string; version: string; sample_rate: number; device: string;
  neural_available: boolean; spectral_available: boolean }
export interface ConfigInfo { models: ModelInfo[]; device: string; configured_sample_rate: number;
  supported_formats: string[]; max_upload_mb: number; max_duration_s: number;
  max_concurrent: number; version: string; rejected_checkpoints: number }
export interface Analysis { waveform: number[]; peak: number[]; spectrogram_db: number[][];
  sample_rate: number; duration_s: number; spectrogram_min_db: number; spectrogram_max_db: number }
export interface ProcessOptions { method: Method; checkpoint: string; strength: number; n_fft: number;
  hop: number; minimum_gain: number; noise_seconds: number; noise_start_seconds: number;
  smooth_time: number; smooth_frequency: number }
