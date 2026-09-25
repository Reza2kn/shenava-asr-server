//! NeMo `AudioToMelSpectrogramPreprocessor`-compatible log-mel fbank, att13 contract.
//!
//! Mirrors the deployed reference (`koochik_server.py` + `preprocessor.json`):
//! sample_rate 16000, n_fft 512, win 400, hop 160, hann (periodic=false), center pad 256
//! reflect, preemphasis 0.97, Slaney 80 mel (80x257), power spectrum (no FFT norm),
//! natural log with guard 2^-24, normalize=NA (NO per-feature normalization).

use anyhow::Result;
use ndarray::{Array1, Array2, Array3};
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use std::io::Cursor;

const N_FFT: usize = 512;
const WIN_LEN: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 80;
const PAD: usize = 256;
const PREEMPH: f32 = 0.97;
const GUARD: f32 = 5.960_464_5e-8;
const FIXED_FRAMES: usize = 2005;
const SAMPLE_RATE: u32 = 16000;

/// Slaney mel filterbank (80 x 257) + hann window + FFT plan.
pub struct Fbank {
    mel: Array2<f32>, // [80, 257]
    window: Vec<f32>, // length N_FFT
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl Fbank {
    pub fn load(mel_filters_path: Option<&str>) -> Result<Self> {
        let mel = if let Some(p) = mel_filters_path {
            let txt = std::fs::read_to_string(p)?;
            let v: serde_json::Value = serde_json::from_str(&txt)?;
            let arr = v.get("filters").unwrap_or(&v);
            let rows = arr
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("mel filters not an array"))?;
            let nrows = rows.len();
            let ncols = rows[0].as_array().unwrap().len();
            let mut flat = Vec::with_capacity(nrows * ncols);
            for r in rows {
                for c in r.as_array().unwrap() {
                    flat.push(c.as_f64().unwrap() as f32);
                }
            }
            let mut mel = Array2::from_shape_vec((nrows, ncols), flat)?;
            if mel.shape()[0] != N_MELS {
                mel = mel.reversed_axes();
            }
            mel
        } else {
            // Embedded fallback: generate Slaney mel (not used in production; model ships with JSON).
            crate::fbank::slaney_mel()?
        };

        // Hann window (periodic=false): w[i] = 0.5 - 0.5*cos(2*pi*i/(win-1)), centered in N_FFT.
        let mut window = vec![0.0f32; N_FFT];
        let off = (N_FFT - WIN_LEN) / 2;
        for i in 0..WIN_LEN {
            let v =
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (WIN_LEN as f32 - 1.0)).cos();
            window[off + i] = v;
        }

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        Ok(Fbank { mel, window, fft })
    }

    /// Compute log-mel features for mono PCM at the configured sample rate.
    /// Returns `(features [80, nf], nf)`.
    pub fn process(&self, sig: &[f32], sr: u32) -> Result<(Array2<f32>, usize)> {
        anyhow::ensure!(!sig.is_empty(), "WAV contains no audio samples");
        let sig = resample(sig, sr, SAMPLE_RATE)?;
        let n = sig.len();
        let nf = 1 + n / HOP;

        // preemphasis
        let mut pre = vec![0.0f32; n];
        if n > 0 {
            pre[0] = sig[0];
            for i in 1..n {
                pre[i] = sig[i] - PREEMPH * sig[i - 1];
            }
        }

        // reflect pad
        let padded = reflect_pad(&pre, PAD);

        let mut frames = Array2::<f32>::zeros((nf, N_FFT));
        let mut reals = vec![0.0f32; N_FFT];
        let mut imags = vec![0.0f32; N_FFT];
        let mut complex = vec![Complex32::default(); N_FFT];
        for i in 0..nf {
            let s = i * HOP;
            for j in 0..N_FFT {
                let v = if s + j < padded.len() {
                    padded[s + j]
                } else {
                    0.0
                };
                reals[j] = v * self.window[j];
            }
            for j in 0..N_FFT {
                complex[j] = Complex32::new(reals[j], 0.0);
            }
            self.fft.process(&mut complex);
            for j in 0..N_FFT {
                imags[j] = complex[j].im * complex[j].im + complex[j].re * complex[j].re;
            }
            for j in 0..=N_FFT / 2 {
                frames[[i, j]] = imags[j];
            }
        }

        // power spectrum [nf, 257] x mel [257, 80] -> [nf, 80]
        let mut logmel = Array2::<f32>::zeros((nf, N_MELS));
        for i in 0..nf {
            for m in 0..N_MELS {
                let mut acc = 0.0f32;
                for j in 0..=N_FFT / 2 {
                    acc += frames[[i, j]] * self.mel[[m, j]];
                }
                logmel[[i, m]] = (acc + GUARD).ln();
            }
        }
        // transpose -> [80, nf]
        let feat = logmel.reversed_axes();
        Ok((feat, nf))
    }

    /// Pad/truncate a feature matrix to [80, FIXED_FRAMES] for the tract model input.
    pub fn to_fixed(&self, feat: &Array2<f32>, nf: usize) -> Array3<f32> {
        let mut out = Array3::<f32>::zeros((1, N_MELS, FIXED_FRAMES));
        let c = nf.min(FIXED_FRAMES);
        for i in 0..N_MELS {
            for j in 0..c {
                out[[0, i, j]] = feat[[i, j]];
            }
        }
        out
    }
}

/// Linear-interpolation resampler (16k target). Fine for clean TTS clips.
fn resample(sig: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to {
        return Ok(sig.to_vec());
    }
    if from == 0 {
        anyhow::bail!("bad sample rate 0");
    }
    let ratio = to as f64 / from as f64;
    let out_len = ((sig.len() as f64) * ratio).ceil() as usize;
    let mut out = vec![0.0f32; out_len];
    for (i, sample) in out.iter_mut().enumerate() {
        let pos = i as f64 / ratio;
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        let a = sig[idx.min(sig.len() - 1)];
        let b = sig[(idx + 1).min(sig.len() - 1)];
        *sample = a + (b - a) * frac;
    }
    Ok(out)
}

/// numpy `np.pad(mode="reflect")`: mirrors without repeating the edge.
/// For `[1,2,3]` with pad=2: `[3, 2, 1, 2, 3, 2, 1]`.
fn reflect_pad(sig: &[f32], pad: usize) -> Vec<f32> {
    let n = sig.len();
    let total = n + 2 * pad;
    let mut out = vec![0.0f32; total];
    if n == 0 {
        return out;
    }
    if n == 1 {
        out.fill(sig[0]);
        return out;
    }
    let period = 2 * (n - 1) as isize;
    for (i, value) in out.iter_mut().enumerate() {
        let source = i as isize - pad as isize;
        let folded = source.rem_euclid(period);
        let index = if folded < n as isize {
            folded
        } else {
            period - folded
        } as usize;
        *value = sig[index];
    }
    out
}

/// Slaney mel filterbank generation (fallback if no JSON). Not used in production.
fn slaney_mel() -> Result<Array2<f32>> {
    anyhow::bail!("slaney mel generation not implemented; ship mel_filters.json")
}

#[allow(dead_code)]
fn _hann_win() -> Array1<f32> {
    Array1::from_iter((0..WIN_LEN).map(|i| {
        0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (WIN_LEN as f32 - 1.0)).cos()
    }))
}

/// Decode a mono WAV file (any bit depth) to f32 PCM + sample rate.
pub fn read_wav_bytes(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let rdr = hound::WavReader::new(Cursor::new(bytes))?;
    let spec = rdr.spec();
    let sr = spec.sample_rate;
    let bits = spec.bits_per_sample;
    let ch = spec.channels as usize;
    anyhow::ensure!(ch > 0, "WAV channel count is zero");
    let samples = match (spec.sample_format, bits) {
        (hound::SampleFormat::Int, 8..=16) => {
            let mut v = Vec::new();
            for s in rdr.into_samples::<i16>() {
                v.push(s? as f32 / (1_u32 << (bits - 1)) as f32);
            }
            v
        }
        (hound::SampleFormat::Int, 17..=32) => {
            let mut v = Vec::new();
            for s in rdr.into_samples::<i32>() {
                v.push(s? as f32 / (1_u64 << (bits - 1)) as f32);
            }
            v
        }
        (hound::SampleFormat::Float, 32) => {
            let mut v = Vec::new();
            for s in rdr.into_samples::<f32>() {
                v.push(s?);
            }
            v
        }
        (format, _) => anyhow::bail!("unsupported WAV sample format: {format:?} {bits}-bit"),
    };
    // mix to mono
    let mono = if ch > 1 {
        let frames = samples.len() / ch;
        let mut m = Vec::with_capacity(frames);
        for f in 0..frames {
            let mut acc = 0.0f32;
            for c in 0..ch {
                acc += samples[f * ch + c];
            }
            m.push(acc / ch as f32);
        }
        m
    } else {
        samples
    };
    Ok((mono, sr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_padding_handles_short_audio() {
        assert_eq!(
            reflect_pad(&[1.0, 2.0, 3.0], 2),
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]
        );
        assert_eq!(reflect_pad(&[4.0], 3), vec![4.0; 7]);
    }
}

/// Bounded PCM window. Two retained hops preserve preemphasis and centered STFT
/// context; future context is held until available, and reflected only at EOF.
#[cfg(any(feature = "native-streaming", test))]
#[derive(Default)]
pub struct LiveFeatures {
    pcm: Vec<f32>,
    base: usize,
    next: usize,
    total: usize,
}

#[cfg(any(feature = "native-streaming", test))]
impl LiveFeatures {
    pub fn push(&mut self, pcm: &[f32]) {
        self.pcm.extend_from_slice(pcm);
        self.total += pcm.len();
    }

    pub fn next_chunk(
        &mut self,
        bank: &Fbank,
        finish: bool,
    ) -> Result<Option<(Array3<f32>, usize)>> {
        const CHUNK: usize = 121;
        const SHIFT: usize = 112;
        if self.total == 0 || self.next >= 1 + self.total / HOP {
            return Ok(None);
        }
        if !finish && self.next * HOP + (CHUNK - 1) * HOP + PAD > self.total {
            return Ok(None);
        }
        let (feat, _) = bank.process(&self.pcm, SAMPLE_RATE)?;
        let offset = self.next - self.base / HOP;
        let valid = CHUNK.min(1 + self.total / HOP - self.next);
        let mut chunk = Array3::zeros((1, N_MELS, CHUNK));
        for m in 0..N_MELS {
            for t in 0..valid {
                chunk[[0, m, t]] = feat[[m, offset + t]];
            }
        }
        self.next += SHIFT;
        let new_base = self.next.saturating_sub(2) * HOP;
        if new_base > self.base && new_base <= self.total {
            self.pcm.drain(..new_base - self.base);
            self.base = new_base;
        }
        Ok(Some((chunk, valid)))
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;

    #[test]
    fn packet_boundaries_match_whole_audio_features_and_final_flush() {
        let bank = Fbank::load(Some(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/mel_filters.json"
        )))
        .unwrap();
        for length in [1, 159, 160, 513, 17920, 19456, 35840, 48017] {
            let audio: Vec<f32> = (0..length)
                .map(|i| (i as f32 * 0.071).sin() * 0.3)
                .collect();
            let (reference, nf) = bank.process(&audio, 16000).unwrap();
            for packet_size in [319, 3200] {
                let mut live = LiveFeatures::default();
                let mut chunks = Vec::new();
                for packet in audio.chunks(packet_size) {
                    live.push(packet);
                    while let Some(chunk) = live.next_chunk(&bank, false).unwrap() {
                        chunks.push(chunk);
                    }
                    assert!(live.pcm.len() < 23000, "PCM storage must remain bounded");
                }
                while let Some(chunk) = live.next_chunk(&bank, true).unwrap() {
                    chunks.push(chunk);
                }
                assert_eq!(chunks.len(), nf.div_ceil(112));
                for (index, (chunk, valid)) in chunks.iter().enumerate() {
                    assert_eq!(*valid, (nf - index * 112).min(121));
                    for m in 0..80 {
                        for t in 0..*valid {
                            assert!((chunk[[0,m,t]] - reference[[m,index*112+t]]).abs() < 1e-5,
                                "feature mismatch length={length}, packet={packet_size}, chunk={index}, mel={m}, frame={t}");
                        }
                    }
                }
            }
        }
    }
}
