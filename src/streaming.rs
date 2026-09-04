//! Native Rust cache-aware streaming CTC inference for Persian Koochik.
//!
//! The graph is exported once from the 114M Koochik checkpoint. Runtime
//! inference is pure Rust through the same Tract fork used by the offline
//! server. The graph consumes already-computed 80-bin NeMo fbank chunks and
//! carries the encoder cache between chunks.

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4};
use tract::prelude::*;

tract::impl_ndarray_interop!();

pub const FEATURE_CHUNK_FRAMES: usize = 121;
pub const FEATURE_SHIFT_FRAMES: usize = 112;
pub const CACHE_LAYERS: usize = 17;
pub const CACHE_LEFT_FRAMES: usize = 70;
pub const CACHE_D_MODEL: usize = 512;
pub const CACHE_TIME_WIDTH: usize = 8;
pub const VOCAB_SIZE: usize = 1025;

pub struct Model {
    runnable: tract::Runnable,
    backend: &'static str,
}

struct CacheState {
    channel: Array4<f32>,
    time: Array4<f32>,
    channel_len: i64,
}

impl Model {
    pub fn load(model_path: &str, backend: crate::model::Backend) -> Result<Self> {
        let backend_name = match backend {
            crate::model::Backend::GpuOrCpu => "gpu-or-cpu",
            crate::model::Backend::Cpu => "cpu",
            crate::model::Backend::CoreMl => {
                anyhow::bail!("CoreML is not supported for the Tract streaming graph")
            }
        };
        let graph = tract::onnx()?.load(model_path)?.into_model()?;
        let runtime = tract::runtime_for_name(backend_name)?;
        log::info!(
            "streaming Tract backend: {} ({} available)",
            backend_name,
            runtime.name().unwrap_or_default()
        );
        let runnable = runtime.prepare(graph)?;
        Ok(Self {
            runnable,
            backend: backend_name,
        })
    }

    pub fn backend(&self) -> &'static str {
        self.backend
    }

    /// Decode an arbitrary-length feature sequence using the graph's cache.
    /// Returns concatenated per-frame CTC log-probabilities.
    pub fn run_features(&self, feat: &Array2<f32>, nf: usize) -> Result<Array2<f32>> {
        anyhow::ensure!(feat.shape()[0] == 80, "streaming fbank must have 80 bins");
        anyhow::ensure!(
            nf > 0 && nf <= feat.shape()[1],
            "invalid streaming frame count"
        );

        let mut state = CacheState {
            channel: Array4::zeros((1, CACHE_LAYERS, CACHE_LEFT_FRAMES, CACHE_D_MODEL)),
            time: Array4::zeros((1, CACHE_LAYERS, CACHE_D_MODEL, CACHE_TIME_WIDTH)),
            channel_len: 0,
        };
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < nf {
            let valid_frames = (nf - start).min(FEATURE_CHUNK_FRAMES);
            let mut feature = Array3::<f32>::zeros((1, 80, FEATURE_CHUNK_FRAMES));
            for mel in 0..80 {
                for frame in 0..valid_frames {
                    feature[[0, mel, frame]] = feat[[mel, start + frame]];
                }
            }

            let length = Array1::<i64>::from_vec(vec![valid_frames as i64]);
            let channel_len = Array1::<i64>::from_vec(vec![state.channel_len]);
            let output = self.runnable.run(vec![
                feature.tract()?,
                length.tract()?,
                state.channel.tract()?,
                state.time.tract()?,
                channel_len.tract()?,
            ])?;

            let logits = output[0].ndarray3::<f32>()?;
            anyhow::ensure!(
                logits.ndim() == 3 && logits.shape()[0] == 1 && logits.shape()[2] == VOCAB_SIZE,
                "streaming logprobs shape is {:?}, expected [1,T,{VOCAB_SIZE}]",
                logits.shape()
            );
            let output_len = output[1].ndarray1::<i64>()?;
            let valid_outputs = (output_len[[0]].max(0) as usize).min(logits.shape()[1]);
            let mut part = Array2::<f32>::zeros((valid_outputs, VOCAB_SIZE));
            for t in 0..valid_outputs {
                for v in 0..VOCAB_SIZE {
                    part[[t, v]] = logits[[0, t, v]];
                }
            }
            chunks.push(part);

            state.channel = output[2].ndarray4::<f32>()?.to_owned();
            state.time = output[3].ndarray4::<f32>()?.to_owned();
            state.channel_len = output[4].ndarray1::<i64>()?[[0]];
            start += FEATURE_SHIFT_FRAMES;
        }

        let total = chunks.iter().map(|v| v.shape()[0]).sum();
        let mut all = Array2::<f32>::zeros((total, VOCAB_SIZE));
        let mut offset = 0;
        for part in chunks {
            let len = part.shape()[0];
            all.slice_mut(ndarray::s![offset..offset + len, ..])
                .assign(&part);
            offset += len;
        }
        Ok(all)
    }
}
