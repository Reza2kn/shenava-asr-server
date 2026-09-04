//! Native Rust runtime for the exported Nemotron-3-Diarization graph.
//!
//! The `.nemo` checkpoint is a NeMo training artifact, not a portable runtime
//! format. The build/deployment boundary is a fixed-shape ONNX export made
//! once with NeMo; inference, feature extraction, streaming state, cache
//! compression, and timestamp ordering all live here in Rust.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3};
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

#[cfg(feature = "tract-runtime")]
use tract::prelude::*;

#[cfg(feature = "tract-runtime")]
tract::impl_ndarray_interop!();

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_MELS: usize = 128;
pub const N_FFT: usize = 512;
pub const HOP: usize = 160;
pub const WIN_LEN: usize = 400;
pub const STACK: usize = 8;
pub const CHUNK_LEN: usize = 340;
pub const RIGHT_CONTEXT: usize = 40;
pub const CHUNK_INPUT_FRAMES: usize = (CHUNK_LEN + RIGHT_CONTEXT) * STACK;
pub const SPKCACHE_LEN: usize = 264;
pub const FIFO_LEN: usize = 40;
pub const EMB_DIM: usize = 512;
pub const SPEAKERS: usize = 8;
pub const FRAME_MS: u64 = 10;
const LOG_GUARD: f32 = 5.960_464_5e-8;
const PREEMPH: f32 = 0.97;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub speaker_id: u32,
}

struct Features {
    /// Time-major `[frames, 128]`, matching the exported graph input.
    frames: Array2<f32>,
}

#[cfg(feature = "tract-runtime")]
pub struct Model {
    runnable: tract::Runnable,
    backend: &'static str,
    silence_embedding: Vec<f32>,
    mel: Array2<f32>,
    window: Vec<f32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

#[cfg(feature = "tract-runtime")]
impl Model {
    pub fn load(model_path: &str, backend: crate::model::Backend) -> Result<Self> {
        let backend_name = match backend {
            crate::model::Backend::GpuOrCpu => "gpu-or-cpu",
            crate::model::Backend::Cpu => "cpu",
            crate::model::Backend::CoreMl => {
                anyhow::bail!("Nemotron native Rust route currently supports Tract CPU/CUDA only")
            }
        };
        let model = tract::onnx()?.load(model_path)?.into_model()?;
        let runtime = tract::runtime_for_name(backend_name)?;
        log::info!(
            "Nemotron Tract backend: {}",
            runtime.name().unwrap_or_default()
        );
        let runnable = runtime.prepare(model)?;

        let silence_path = format!("{model_path}.silence.bin");
        let silence_embedding = match std::fs::read(&silence_path) {
            Ok(bytes) => {
                anyhow::ensure!(
                    bytes.len() == EMB_DIM * 4,
                    "Nemotron silence sidecar has {} bytes, expected {}",
                    bytes.len(),
                    EMB_DIM * 4
                );
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            }
            Err(_) => {
                log::warn!("Nemotron silence sidecar missing; cache padding will use zero embedding: {silence_path}");
                vec![0.0; EMB_DIM]
            }
        };
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        Ok(Self {
            runnable,
            backend: backend_name,
            silence_embedding,
            mel: slaney_mel(SAMPLE_RATE, N_FFT, N_MELS),
            window: hann_window(),
            fft,
        })
    }

    pub fn backend(&self) -> &'static str {
        self.backend
    }

    pub fn diarize_wav(&self, bytes: &[u8]) -> Result<Vec<Segment>> {
        let (signal, sample_rate) = crate::fbank::read_wav_bytes(bytes)?;
        let features = self.features(&signal, sample_rate)?;
        self.diarize_features(&features)
    }

    fn features(&self, signal: &[f32], sample_rate: u32) -> Result<Features> {
        let signal = resample(signal, sample_rate, SAMPLE_RATE)?;
        anyhow::ensure!(!signal.is_empty(), "WAV contains no audio samples");
        let mut pre = vec![0.0; signal.len()];
        pre[0] = signal[0];
        for i in 1..signal.len() {
            pre[i] = signal[i] - PREEMPH * signal[i - 1];
        }

        // NeMo's exact_pad=false uses torch.stft(center=true, pad_mode=constant),
        // hence one zero-padded frame at either side, not reflect padding.
        let padded_len = pre.len() + N_FFT;
        let frame_count = 1 + pre.len() / HOP;
        let mut output = Array2::<f32>::zeros((frame_count, N_MELS));
        let mut spectrum = vec![Complex32::default(); N_FFT];
        for frame in 0..frame_count {
            let start = frame * HOP;
            for j in 0..N_FFT {
                let source = start + j;
                let value = if source < padded_len {
                    let centered = source as isize - (N_FFT / 2) as isize;
                    if centered >= 0 && (centered as usize) < pre.len() {
                        pre[centered as usize]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                spectrum[j] = Complex32::new(value * self.window[j], 0.0);
            }
            self.fft.process(&mut spectrum);
            for mel_bin in 0..N_MELS {
                let mut energy = 0.0;
                for fft_bin in 0..=N_FFT / 2 {
                    let power = spectrum[fft_bin].re * spectrum[fft_bin].re
                        + spectrum[fft_bin].im * spectrum[fft_bin].im;
                    energy += power * self.mel[[mel_bin, fft_bin]];
                }
                output[[frame, mel_bin]] = (energy + LOG_GUARD).ln();
            }
        }
        Ok(Features { frames: output })
    }

    fn diarize_features(&self, features: &Features) -> Result<Vec<Segment>> {
        let mut state = State::new(&self.silence_embedding);
        let mut all_preds: Vec<[f32; SPEAKERS]> = Vec::new();
        let mut start = 0;
        while start < features.frames.nrows() {
            let core_end = (start + CHUNK_LEN * STACK).min(features.frames.nrows());
            let right = (RIGHT_CONTEXT * STACK).min(features.frames.nrows() - core_end);
            let input_end = core_end + right;
            let raw_len = input_end - start;
            let emb_len = raw_len.div_ceil(STACK);
            let core_emb_len = emb_len.saturating_sub(right.div_ceil(STACK));

            let mut chunk = Array3::<f32>::zeros((1, CHUNK_INPUT_FRAMES, N_MELS));
            for t in 0..raw_len {
                for d in 0..N_MELS {
                    chunk[[0, t, d]] = features.frames[[start + t, d]];
                }
            }
            let (state_preds, highres_preds, embeddings) =
                self.run_chunk(&chunk, raw_len, &state)?;
            let base = state.cache_len + state.fifo_len;
            let highres_base = base * STACK;
            for t in 0..core_emb_len {
                for subframe in 0..STACK {
                    let mut row = [0.0; SPEAKERS];
                    for s in 0..SPEAKERS {
                        row[s] = highres_preds[[highres_base + t * STACK + subframe, s]];
                    }
                    all_preds.push(row);
                }
            }
            state.update(
                &embeddings[..core_emb_len * EMB_DIM],
                &state_preds,
                core_emb_len,
                &self.silence_embedding,
            );
            start = core_end;
        }
        Ok(predictions_to_segments(&all_preds))
    }

    fn run_chunk(
        &self,
        chunk: &Array3<f32>,
        raw_len: usize,
        state: &State,
    ) -> Result<(Array2<f32>, Array2<f32>, Vec<f32>)> {
        let chunk_length = Array1::<i64>::from_vec(vec![raw_len as i64]).tract()?;
        let cache_length = Array1::<i64>::from_vec(vec![state.cache_len as i64]).tract()?;
        let fifo_length = Array1::<i64>::from_vec(vec![state.fifo_len as i64]).tract()?;
        let mut cache = Array3::<f32>::zeros((1, SPKCACHE_LEN, EMB_DIM));
        for t in 0..state.cache_len {
            for d in 0..EMB_DIM {
                cache[[0, t, d]] = state.cache[t * EMB_DIM + d];
            }
        }
        let mut fifo = Array3::<f32>::zeros((1, FIFO_LEN, EMB_DIM));
        for t in 0..state.fifo_len {
            for d in 0..EMB_DIM {
                fifo[[0, t, d]] = state.fifo[t * EMB_DIM + d];
            }
        }
        let outputs = self.runnable.run(vec![
            chunk.clone().tract()?,
            chunk_length,
            cache.tract()?,
            cache_length,
            fifo.tract()?,
            fifo_length,
        ])?;
        let preds = outputs[0].ndarray3::<f32>()?;
        let pred_len = preds.shape()[1];
        let mut pred_copy = Array2::<f32>::zeros((pred_len, SPEAKERS));
        for t in 0..pred_len {
            for s in 0..SPEAKERS {
                pred_copy[[t, s]] = preds[[0, t, s]];
            }
        }
        let highres = outputs[3].ndarray3::<f32>()?;
        let highres_len = highres.shape()[1];
        let mut highres_copy = Array2::<f32>::zeros((highres_len, SPEAKERS));
        for t in 0..highres_len {
            for s in 0..SPEAKERS {
                highres_copy[[t, s]] = highres[[0, t, s]];
            }
        }
        let embeddings = outputs[1]
            .as_slice::<f32>()
            .context("Nemotron chunk embeddings are not f32")?
            .to_vec();
        let actual_emb_len = outputs[2]
            .as_slice::<i64>()
            .context("Nemotron embedding length is not int64")?
            .first()
            .copied()
            .unwrap_or(0)
            .max(0) as usize;
        anyhow::ensure!(
            actual_emb_len * EMB_DIM <= embeddings.len(),
            "Nemotron embedding length exceeds output"
        );
        Ok((pred_copy, highres_copy, embeddings))
    }
}

#[derive(Default)]
struct State {
    cache: Vec<f32>,
    cache_preds: Vec<[f32; SPEAKERS]>,
    cache_len: usize,
    fifo: Vec<f32>,
    fifo_preds: Vec<[f32; SPEAKERS]>,
    fifo_len: usize,
    compressed: bool,
}

impl State {
    fn new(_silence: &[f32]) -> Self {
        Self::default()
    }

    fn update(&mut self, chunk: &[f32], preds: &Array2<f32>, chunk_len: usize, silence: &[f32]) {
        let old_cache = self.cache_len;
        let old_fifo = self.fifo_len;
        let mut current_fifo_preds = Vec::with_capacity(old_fifo);
        for t in 0..old_fifo {
            current_fifo_preds.push(row(preds, old_cache + t));
        }
        let mut current_chunk_preds = Vec::with_capacity(chunk_len);
        for t in 0..chunk_len {
            current_chunk_preds.push(row(preds, old_cache + old_fifo + t));
        }
        // NeMo refreshes FIFO predictions from the current forward pass before
        // appending the new chunk. Cache predictions are refreshed on the
        // next compression boundary in the same way.
        self.fifo_preds = current_fifo_preds;
        self.fifo.extend_from_slice(chunk);
        self.fifo_preds.extend(current_chunk_preds);
        self.fifo_len += chunk_len;
        if old_fifo + chunk_len <= FIFO_LEN {
            return;
        }

        let pop_len = 300
            .max(chunk_len.saturating_sub(FIFO_LEN).saturating_add(old_fifo))
            .min(self.fifo_len);
        let popped_embs: Vec<f32> = self.fifo.drain(..pop_len * EMB_DIM).collect();
        let popped_preds: Vec<[f32; SPEAKERS]> = self.fifo_preds.drain(..pop_len).collect();
        self.fifo_len -= pop_len;
        if !self.compressed {
            let mut fresh = Vec::with_capacity(old_cache + pop_len);
            fresh.extend(self.cache_preds.iter().copied());
            fresh.extend(popped_preds.iter().copied());
            self.cache_preds = fresh;
        } else {
            self.cache_preds.extend(popped_preds.iter().copied());
        }
        self.cache.extend_from_slice(&popped_embs);
        self.cache_len += pop_len;
        if self.cache_len > SPKCACHE_LEN {
            self.compress(silence);
        }
    }

    fn compress(&mut self, silence: &[f32]) {
        let n = self.cache_len;
        let per_spk = SPKCACHE_LEN / SPEAKERS - 1;
        let strong = (per_spk as f32 * 0.75).floor() as usize;
        let weak = (per_spk as f32 * 1.5).floor() as usize;
        let min_pos = (per_spk as f32 * 0.5).floor() as usize;
        let mut scores = vec![f32::NEG_INFINITY; n * SPEAKERS];
        for t in 0..n {
            let p = self.cache_preds[t];
            let mut sum_log_one = 0.0;
            for value in p {
                sum_log_one += value.clamp(0.25, 0.75).ln();
            }
            for s in 0..SPEAKERS {
                let logit = p[s].clamp(0.25, 0.75).ln() - (1.0 - p[s]).clamp(0.25, 0.75).ln()
                    + sum_log_one
                    - 0.5f32.ln();
                if p[s] > 0.5 {
                    scores[s * n + t] = logit + if t >= SPKCACHE_LEN { 0.05 } else { 0.0 };
                }
            }
        }
        // Disable non-positive overlapping scores when a speaker has enough
        // positive evidence, matching SortformerModules._disable_low_scores.
        for s in 0..SPEAKERS {
            let positives = (0..n).filter(|&t| scores[s * n + t] > 0.0).count();
            if positives >= min_pos {
                for t in 0..n {
                    if self.cache_preds[t][s] > 0.5 && scores[s * n + t] <= 0.0 {
                        scores[s * n + t] = f32::NEG_INFINITY;
                    }
                }
            }
            boost_topk(&mut scores[s * n..(s + 1) * n], strong, 2.0);
            boost_topk(&mut scores[s * n..(s + 1) * n], weak, 1.0);
        }
        // One synthetic silence frame is added for each speaker block. Its
        // eight +inf entries guarantee one learned silence slot per speaker.
        let mut candidates: Vec<(f32, usize, usize)> = Vec::with_capacity(n * SPEAKERS + SPEAKERS);
        for s in 0..SPEAKERS {
            for t in 0..n {
                candidates.push((scores[s * n + t], t, s));
            }
            candidates.push((f32::INFINITY, n, s));
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut chosen = candidates
            .into_iter()
            .take(SPKCACHE_LEN)
            .collect::<Vec<_>>();
        // NeMo sorts flattened speaker-major indices: each speaker's chosen
        // frames stay together, and frames within that speaker stay in time
        // order. This is part of the speaker-ID stability contract.
        chosen.sort_by_key(|(_, t, s)| (*s, *t));
        let mut next_cache = Vec::with_capacity(SPKCACHE_LEN * EMB_DIM);
        let mut next_preds = Vec::with_capacity(SPKCACHE_LEN);
        for (_, t, s) in chosen {
            if t >= n {
                next_cache.extend_from_slice(silence);
                next_preds.push([0.0; SPEAKERS]);
            } else {
                next_cache.extend_from_slice(&self.cache[t * EMB_DIM..(t + 1) * EMB_DIM]);
                next_preds.push(self.cache_preds[t]);
            }
            let _ = s;
        }
        self.cache = next_cache;
        self.cache_preds = next_preds;
        self.cache_len = SPKCACHE_LEN;
        self.compressed = true;
    }
}

fn row(preds: &Array2<f32>, t: usize) -> [f32; SPEAKERS] {
    let mut out = [0.0; SPEAKERS];
    for s in 0..SPEAKERS {
        out[s] = preds[[t, s]];
    }
    out
}

fn boost_topk(scores: &mut [f32], k: usize, scale: f32) {
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    for i in indices.into_iter().take(k) {
        scores[i] -= scale * 0.5f32.ln();
    }
}

fn predictions_to_segments(preds: &[[f32; SPEAKERS]]) -> Vec<Segment> {
    let mut result = Vec::new();
    for speaker in 0..SPEAKERS {
        let mut start = None;
        for (frame, row) in preds.iter().enumerate() {
            let active = row[speaker] >= 0.5;
            match (start, active) {
                (None, true) => start = Some(frame),
                (Some(begin), false) => {
                    result.push(Segment {
                        start_ms: begin as u64 * FRAME_MS,
                        end_ms: frame as u64 * FRAME_MS,
                        speaker_id: speaker as u32,
                    });
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(begin) = start {
            result.push(Segment {
                start_ms: begin as u64 * FRAME_MS,
                end_ms: preds.len() as u64 * FRAME_MS,
                speaker_id: speaker as u32,
            });
        }
    }
    // This is intentionally a flat event stream: same-time segments remain
    // adjacent, so overlap is visible while the audio order is retained.
    result.sort_by_key(|s| (s.start_ms, s.end_ms, s.speaker_id));
    result
}

fn hann_window() -> Vec<f32> {
    let mut out = vec![0.0; N_FFT];
    let offset = (N_FFT - WIN_LEN) / 2;
    for i in 0..WIN_LEN {
        out[offset + i] =
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (WIN_LEN as f32 - 1.0)).cos();
    }
    out
}

fn slaney_mel(sr: u32, n_fft: usize, n_mels: usize) -> Array2<f32> {
    let fmax = sr as f64 / 2.0;
    let hz_to_mel = |hz: f64| {
        if hz >= 1000.0 {
            1000.0 / (200.0 / 3.0) + (hz / 1000.0).ln() / (6.4f64.ln() / 27.0)
        } else {
            hz / (200.0 / 3.0)
        }
    };
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(fmax);
    let mut centers = Vec::with_capacity(n_mels + 2);
    for i in 0..n_mels + 2 {
        let mel = mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64;
        centers.push(if mel >= 1000.0 / (200.0 / 3.0) {
            1000.0 * ((mel - 1000.0 / (200.0 / 3.0)) * (6.4f64.ln() / 27.0)).exp()
        } else {
            mel * (200.0 / 3.0)
        });
    }
    let mut out = Array2::<f32>::zeros((n_mels, n_fft / 2 + 1));
    for m in 0..n_mels {
        let left = centers[m];
        let center = centers[m + 1];
        let right = centers[m + 2];
        let left_width = center - left;
        let right_width = right - center;
        for k in 0..=n_fft / 2 {
            let hz = sr as f64 * k as f64 / n_fft as f64;
            let lower = (hz - left) / left_width;
            let upper = (right - hz) / right_width;
            out[[m, k]] =
                0.0f64.max(lower.min(upper)).max(0.0) as f32 * (2.0 / (right - left)) as f32;
        }
    }
    out
}

fn resample(signal: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to {
        return Ok(signal.to_vec());
    }
    anyhow::ensure!(from > 0, "bad sample rate 0");
    let ratio = to as f64 / from as f64;
    let out_len = (signal.len() as f64 * ratio).ceil() as usize;
    let mut out = vec![0.0; out_len];
    for (i, value) in out.iter_mut().enumerate() {
        let pos = i as f64 / ratio;
        let index = pos.floor() as usize;
        let frac = (pos - index as f64) as f32;
        let a = signal[index.min(signal.len() - 1)];
        let b = signal[(index + 1).min(signal.len() - 1)];
        *value = a + (b - a) * frac;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_has_expected_shape_and_positive_bands() {
        let mel = slaney_mel(SAMPLE_RATE, N_FFT, N_MELS);
        assert_eq!(mel.shape(), &[N_MELS, N_FFT / 2 + 1]);
        assert!(mel.iter().any(|x| *x > 0.0));
    }

    #[test]
    fn segments_are_audio_ordered_and_keep_overlap() {
        let mut p = vec![[0.0; SPEAKERS]; 4];
        p[0][1] = 1.0;
        p[1][0] = 1.0;
        p[1][1] = 1.0;
        let got = predictions_to_segments(&p);
        assert_eq!(got[0].start_ms, 0);
        assert_eq!(got[0].speaker_id, 1);
        assert_eq!(got[1].start_ms, 10);
        assert_eq!(got[1].speaker_id, 0);
    }
}
