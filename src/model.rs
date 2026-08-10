//! tract model wrapper for the Shenava Koochik offline ONNX.
//!
//! Loads the pre-simplified `model.onnx` from `Reza2kn/Shenava-Koochik-v1.0-tract-offline`
//! (fixed `[1, 80, 2005]` input) and runs it, returning log_probs `[T, V]` for the valid
//! output frames.
//!
//! Backend selection is pluggable: pass `--backend gpu-or-cpu` (default) to use tract's
//! `gpu-or-cpu` runtime, which resolves to the first *available* accelerator — Metal on
//! Apple silicon, CUDA on NVIDIA — and falls back to CPU. Pass `--backend cpu` to force
//! CPU. New backends (ANE, MPS, …) can be added by extending the runtime names tried.

use anyhow::Result;
use ndarray::Array2;
use tract::prelude::*;

tract::impl_ndarray_interop!();

pub const INPUT_FRAMES: usize = 2005;

/// Which tract runtime to prepare the model on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    GpuOrCpu,
    Cpu,
}

impl Backend {
    fn runtime_name(self) -> &'static str {
        match self {
            Backend::GpuOrCpu => "gpu-or-cpu",
            Backend::Cpu => "cpu",
        }
    }
}

pub struct KoochikModel {
    runnable: tract::Runnable,
    backend: &'static str,
}

impl KoochikModel {
    /// Load a pre-built tract model from an ONNX path on the given backend.
    pub fn load(model_path: &str, backend: Backend) -> Result<Self> {
        let name = backend.runtime_name();
        let model = tract::onnx()?.load(model_path)?.into_model()?;
        let runtime = tract::runtime_for_name(name)?;
        log::info!("tract backend: {} ({} available)", name, runtime.name().unwrap_or_default());
        let runnable = runtime.prepare(model)?;
        Ok(KoochikModel { runnable, backend: name })
    }

    /// Run fbank features (already fixed `[1, 80, 2005]`) + valid frame count.
    /// Returns `(log_probs [T, 1025], output_length)`.
    pub fn run(
        &self,
        feat_fixed: &ndarray::Array3<f32>,
        nf: usize,
    ) -> Result<(Array2<f32>, usize)> {
        let length = ndarray::Array1::<i64>::from(vec![nf.min(INPUT_FRAMES) as i64]);
        let inputs: Vec<tract::Tensor> = vec![feat_fixed.clone().tract()?, length.tract()?];
        let out = self.runnable.run(inputs)?;
        // outputs: log_probs [1, T', 1025], output_length [1]
        let lp = out[0].ndarray3::<f32>()?;
        let t_dim = lp.shape()[1];
        let v_dim = lp.shape()[2];
        let ol = if out.len() > 1 {
            let ol_arr = out[1].ndarray1::<i64>()?;
            ol_arr[[0]] as usize
        } else {
            t_dim
        };
        let valid = ol.min(t_dim);
        let mut arr = Array2::<f32>::zeros((valid, v_dim));
        for t in 0..valid {
            for v in 0..v_dim {
                arr[[t, v]] = lp[[0, t, v]];
            }
        }
        Ok((arr, valid))
    }
}
