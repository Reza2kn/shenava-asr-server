//! Native Rust cache-aware streaming CTC inference for Persian Koochik.
//!
//! The graph is exported once from the 114M Koochik checkpoint. Runtime
//! inference is pure Rust through the same Tract fork used by the offline
//! server. The graph consumes already-computed 80-bin NeMo fbank chunks and
//! carries the encoder cache between chunks.

use anyhow::Result;
#[cfg(feature = "tract-runtime")]
use ndarray::Array1;
use ndarray::{Array2, Array3, Array4};
#[cfg(feature = "tract-runtime")]
use tract::prelude::*;

#[cfg(feature = "tract-runtime")]
tract::impl_ndarray_interop!();

pub const FEATURE_CHUNK_FRAMES: usize = 121;
pub const FEATURE_SHIFT_FRAMES: usize = 112;
pub const CACHE_LAYERS: usize = 17;
pub const CACHE_LEFT_FRAMES: usize = 70;
pub const CACHE_D_MODEL: usize = 512;
pub const CACHE_TIME_WIDTH: usize = 8;
pub const VOCAB_SIZE: usize = 1025;

pub struct Model {
    runnable: Execution,
    backend: &'static str,
}

enum Execution {
    Cpu(std::sync::Arc<tract_onnx::prelude::TypedRunnableModel>),
    #[cfg(feature = "tract-runtime")]
    Runtime(tract::Runnable),
}

pub struct CacheState {
    channel: Array4<f32>,
    time: Array4<f32>,
    channel_len: i64,
}

impl Model {
    pub fn load(model_path: &str, backend: crate::model::Backend) -> Result<Self> {
        if backend == crate::model::Backend::Cpu
            || (backend == crate::model::Backend::GpuOrCpu && !cfg!(feature = "tract-runtime"))
        {
            use tract_onnx::prelude::*;
            let runnable = tract_onnx::onnx()
                .model_for_path(model_path)?
                .into_typed()?
                .into_decluttered()?
                .into_runnable()?;
            return Ok(Self {
                runnable: Execution::Cpu(runnable),
                backend: "cpu",
            });
        }
        #[cfg(not(feature = "tract-runtime"))]
        anyhow::bail!(
            "This streaming build supports CPU; CUDA requires --features cuda,native-streaming"
        );
        #[cfg(feature = "tract-runtime")]
        {
            let backend_name = match backend {
                crate::model::Backend::Cuda => "cuda",
                crate::model::Backend::GpuOrCpu => "gpu-or-cpu",
                crate::model::Backend::Cpu => "cpu",
                crate::model::Backend::CoreMl => {
                    anyhow::bail!("CoreML is not supported for the Tract streaming graph")
                }
            };
            let graph = tract::onnx()?.load(model_path)?.into_model()?;
            let runtime = tract::runtime_for_name(backend_name)?;
            let runtime_name = runtime.name().unwrap_or_else(|_| backend_name.to_owned());
            let actual_backend = match runtime_name.as_str() {
                "cuda" => "cuda",
                "metal" => "metal",
                "cpu" => "cpu",
                _ => backend_name,
            };
            log::info!(
                "streaming Tract backend: {} ({} available)",
                backend_name,
                actual_backend
            );
            let runnable = runtime.prepare(graph)?;
            Ok(Self {
                runnable: Execution::Runtime(runnable),
                backend: actual_backend,
            })
        }
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

        let mut state = CacheState::new();
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

            chunks.push(self.run_chunk(feature, valid_frames, &mut state)?);
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
    /// Advance one connection's encoder cache; weights are shared across connections.
    pub fn run_chunk(
        &self,
        feature: Array3<f32>,
        valid_frames: usize,
        state: &mut CacheState,
    ) -> Result<Array2<f32>> {
        anyhow::ensure!(
            valid_frames > 0 && valid_frames <= FEATURE_CHUNK_FRAMES,
            "invalid chunk length"
        );
        match &self.runnable {
            Execution::Cpu(runnable) => {
                use tract_onnx::prelude::*;
                let inputs = tvec!(
                    Tensor::from_shape(feature.shape(), feature.as_slice().unwrap())?.into_tvalue(),
                    Tensor::from_shape(&[1], &[valid_frames as i64])?.into_tvalue(),
                    Tensor::from_shape(state.channel.shape(), state.channel.as_slice().unwrap())?
                        .into_tvalue(),
                    Tensor::from_shape(state.time.shape(), state.time.as_slice().unwrap())?
                        .into_tvalue(),
                    Tensor::from_shape(&[1], &[state.channel_len])?.into_tvalue(),
                );
                let output = runnable.run(inputs)?;
                let logits = output[0].to_plain_array_view::<f32>()?;
                anyhow::ensure!(
                    logits.ndim() == 3 && logits.shape()[0] == 1 && logits.shape()[2] == VOCAB_SIZE,
                    "unexpected streaming logits shape"
                );
                let count = (output[1].to_plain_array_view::<i64>()?[[0]].max(0) as usize)
                    .min(logits.shape()[1]);
                let part = Array2::from_shape_fn((count, VOCAB_SIZE), |(t, v)| logits[[0, t, v]]);
                let channel = output[2].to_plain_array_view::<f32>()?;
                let time = output[3].to_plain_array_view::<f32>()?;
                state.channel = Array4::from_shape_fn(
                    (1, CACHE_LAYERS, CACHE_LEFT_FRAMES, CACHE_D_MODEL),
                    |(a, b, c, d)| channel[[a, b, c, d]],
                );
                state.time = Array4::from_shape_fn(
                    (1, CACHE_LAYERS, CACHE_D_MODEL, CACHE_TIME_WIDTH),
                    |(a, b, c, d)| time[[a, b, c, d]],
                );
                state.channel_len = output[4].to_plain_array_view::<i64>()?[[0]];
                Ok(part)
            }
            #[cfg(feature = "tract-runtime")]
            Execution::Runtime(runnable) => {
                let length = Array1::<i64>::from_vec(vec![valid_frames as i64]);
                let channel_len = Array1::<i64>::from_vec(vec![state.channel_len]);
                let output = runnable.run(vec![
                    feature.tract()?,
                    length.tract()?,
                    std::mem::take(&mut state.channel).tract()?,
                    std::mem::take(&mut state.time).tract()?,
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

                state.channel = output[2].ndarray4::<f32>()?.to_owned();
                state.time = output[3].ndarray4::<f32>()?.to_owned();
                state.channel_len = output[4].ndarray1::<i64>()?[[0]];
                Ok(part)
            }
        }
    }
}

impl CacheState {
    pub fn new() -> Self {
        Self {
            channel: Array4::zeros((1, CACHE_LAYERS, CACHE_LEFT_FRAMES, CACHE_D_MODEL)),
            time: Array4::zeros((1, CACHE_LAYERS, CACHE_D_MODEL, CACHE_TIME_WIDTH)),
            channel_len: 0,
        }
    }
}
