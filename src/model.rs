//! Inference backends for Shenava Koochik.
//!
//! Both implementations consume the same fixed `[1, 80, 2005]` fbank tensor
//! and return CTC scores shaped `[T, 1025]`. That keeps audio preprocessing and
//! decoding identical across CPU/CUDA/Metal tract and native Apple CoreML.

use anyhow::{Context, Result};
use ndarray::Array2;
#[cfg(feature = "tract-runtime")]
use tract::prelude::*;

#[cfg(feature = "tract-runtime")]
tract::impl_ndarray_interop!();

pub const INPUT_FRAMES: usize = 2005;
const VOCAB_SIZE: usize = 1025;

/// Runtime selected for acoustic-model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    /// tract auto-selection: CUDA/Metal when available, otherwise CPU.
    #[value(name = "gpu-or-cpu", alias = "auto")]
    GpuOrCpu,
    /// Pure-Rust tract CPU runtime. This is the Windows CPU-only target.
    #[value(name = "cpu", alias = "tract-cpu")]
    Cpu,
    /// Native CoreML (CPU/GPU/ANE), available with `--features coreml` on Apple.
    #[value(name = "coreml")]
    CoreMl,
}

#[cfg(feature = "tract-runtime")]
impl Backend {
    fn tract_runtime_name(self) -> Option<&'static str> {
        match self {
            Backend::GpuOrCpu => Some("gpu-or-cpu"),
            Backend::Cpu => Some("cpu"),
            Backend::CoreMl => None,
        }
    }
}

/// One model contract with platform-specific implementations behind it.
pub enum InferenceModel {
    #[cfg(feature = "tract-runtime")]
    Tract(TractModel),
    #[cfg(feature = "cpu-only")]
    DirectCpu(DirectCpuModel),
    #[cfg(all(feature = "coreml", target_vendor = "apple"))]
    CoreMl(CoreMlModel),
}

impl InferenceModel {
    pub fn load(model_path: &str, backend: Backend) -> Result<Self> {
        match backend {
            Backend::GpuOrCpu => Self::load_tract(model_path, backend),
            Backend::Cpu => Self::load_cpu(model_path),
            Backend::CoreMl => Self::load_coreml(model_path),
        }
    }

    #[cfg(feature = "tract-runtime")]
    fn load_tract(model_path: &str, backend: Backend) -> Result<Self> {
        Ok(Self::Tract(TractModel::load(model_path, backend)?))
    }

    #[cfg(not(feature = "tract-runtime"))]
    fn load_tract(_model_path: &str, _backend: Backend) -> Result<Self> {
        anyhow::bail!("gpu-or-cpu requires the `tract-runtime` Cargo feature")
    }

    #[cfg(feature = "cpu-only")]
    fn load_cpu(model_path: &str) -> Result<Self> {
        Ok(Self::DirectCpu(DirectCpuModel::load(model_path)?))
    }

    #[cfg(all(not(feature = "cpu-only"), feature = "tract-runtime"))]
    fn load_cpu(model_path: &str) -> Result<Self> {
        Self::load_tract(model_path, Backend::Cpu)
    }

    #[cfg(not(any(feature = "cpu-only", feature = "tract-runtime")))]
    fn load_cpu(_model_path: &str) -> Result<Self> {
        anyhow::bail!("CPU inference requires `tract-runtime` or `cpu-only`")
    }

    #[cfg(all(feature = "coreml", target_vendor = "apple"))]
    fn load_coreml(model_path: &str) -> Result<Self> {
        Ok(Self::CoreMl(CoreMlModel::load(model_path)?))
    }

    #[cfg(not(all(feature = "coreml", target_vendor = "apple")))]
    fn load_coreml(_model_path: &str) -> Result<Self> {
        anyhow::bail!("CoreML requires an Apple target and a build made with `--features coreml`")
    }

    pub fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "tract-runtime")]
            Self::Tract(model) => model.backend,
            #[cfg(feature = "cpu-only")]
            Self::DirectCpu(_) => "cpu",
            #[cfg(all(feature = "coreml", target_vendor = "apple"))]
            Self::CoreMl(_) => "coreml",
        }
    }

    pub fn run(
        &self,
        feat_fixed: &ndarray::Array3<f32>,
        nf: usize,
    ) -> Result<(Array2<f32>, usize)> {
        match self {
            #[cfg(feature = "tract-runtime")]
            Self::Tract(model) => model.run(feat_fixed, nf),
            #[cfg(feature = "cpu-only")]
            Self::DirectCpu(model) => model.run(feat_fixed, nf),
            #[cfg(all(feature = "coreml", target_vendor = "apple"))]
            Self::CoreMl(model) => model.run(feat_fixed, nf),
        }
    }
}

#[cfg(feature = "tract-runtime")]
pub struct TractModel {
    runnable: tract::Runnable,
    backend: &'static str,
}

#[cfg(feature = "tract-runtime")]
impl TractModel {
    fn load(model_path: &str, backend: Backend) -> Result<Self> {
        let name = backend
            .tract_runtime_name()
            .context("CoreML is not a tract runtime")?;
        let model = tract::onnx()?.load(model_path)?.into_model()?;
        let runtime = tract::runtime_for_name(name)?;
        log::info!(
            "tract backend: {} ({} available)",
            name,
            runtime.name().unwrap_or_default()
        );
        let runnable = runtime.prepare(model)?;
        Ok(Self {
            runnable,
            backend: name,
        })
    }

    fn run(&self, feat_fixed: &ndarray::Array3<f32>, nf: usize) -> Result<(Array2<f32>, usize)> {
        let length = ndarray::Array1::<i64>::from(vec![nf.min(INPUT_FRAMES) as i64]);
        let inputs: Vec<tract::Tensor> = vec![feat_fixed.clone().tract()?, length.tract()?];
        let out = self.runnable.run(inputs)?;
        let lp = out[0].ndarray3::<f32>()?;
        let t_dim = lp.shape()[1];
        let v_dim = lp.shape()[2];
        anyhow::ensure!(
            v_dim == VOCAB_SIZE,
            "model vocabulary is {v_dim}, expected {VOCAB_SIZE}"
        );
        let ol = if out.len() > 1 {
            out[1].ndarray1::<i64>()?[[0]] as usize
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

/// Direct tract ONNX execution with only the default CPU runtime linked. This
/// is separate from the runtime registry because the Shenava tract fork links
/// its CUDA crate at the target level on Windows even when CUDA is disabled.
#[cfg(feature = "cpu-only")]
pub struct DirectCpuModel {
    runnable: std::sync::Arc<tract_onnx::prelude::TypedRunnableModel>,
}

#[cfg(feature = "cpu-only")]
impl DirectCpuModel {
    fn load(model_path: &str) -> Result<Self> {
        use tract_onnx::prelude::*;

        let runnable = tract_onnx::onnx()
            .model_for_path(model_path)?
            .into_typed()?
            .into_decluttered()?
            .into_runnable()?;
        log::info!("tract backend: cpu-only");
        Ok(Self { runnable })
    }

    fn run(&self, feat_fixed: &ndarray::Array3<f32>, nf: usize) -> Result<(Array2<f32>, usize)> {
        use tract_onnx::prelude::*;

        let feature_data = feat_fixed
            .as_slice()
            .context("tract CPU input features must be contiguous")?;
        let feature_tensor = Tensor::from_shape(&[1, 80, INPUT_FRAMES], feature_data)?;
        let length_data = [nf.min(INPUT_FRAMES) as i64];
        let length_tensor = Tensor::from_shape(&[1], &length_data)?;
        let out = self.runnable.run(tvec!(
            feature_tensor.into_tvalue(),
            length_tensor.into_tvalue()
        ))?;
        let lp = out[0].to_plain_array_view::<f32>()?;
        anyhow::ensure!(
            lp.ndim() == 3,
            "CPU logits rank is {}, expected 3",
            lp.ndim()
        );
        let t_dim = lp.shape()[1];
        let v_dim = lp.shape()[2];
        anyhow::ensure!(
            v_dim == VOCAB_SIZE,
            "model vocabulary is {v_dim}, expected {VOCAB_SIZE}"
        );
        let ol = if out.len() > 1 {
            out[1].to_plain_array_view::<i64>()?[[0]] as usize
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

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
pub struct CoreMlModel {
    model: coreml_native::Model,
    // CoreML puts runtime-compiled packages in a temporary directory. Keep the
    // path for diagnostics and to make that lifecycle explicit.
    _compiled_path: Option<std::path::PathBuf>,
}

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
impl CoreMlModel {
    fn load(model_path: &str) -> Result<Self> {
        use coreml_native::{compile_model, ComputeUnits, Model};
        use std::path::Path;

        let source = Path::new(model_path);
        anyhow::ensure!(source.exists(), "CoreML model does not exist: {model_path}");
        let is_compiled = source.extension().and_then(|v| v.to_str()) == Some("mlmodelc");
        let (load_path, compiled_path) = if is_compiled {
            (source.to_path_buf(), None)
        } else {
            log::info!("compiling CoreML package: {model_path}");
            let path = compile_model(source).context("compile CoreML model")?;
            (path.clone(), Some(path))
        };

        let model = Model::load(&load_path, ComputeUnits::All).context("load CoreML model")?;
        validate_coreml_contract(&model)?;
        log::info!("CoreML model loaded with CPU/GPU/ANE compute units");
        Ok(Self {
            model,
            _compiled_path: compiled_path,
        })
    }

    fn run(&self, feat_fixed: &ndarray::Array3<f32>, nf: usize) -> Result<(Array2<f32>, usize)> {
        use coreml_native::{AsMultiArray, BorrowedTensor};

        let feature_data = feat_fixed
            .as_slice()
            .context("CoreML input features must be contiguous")?;
        let length_data = [nf.min(INPUT_FRAMES) as i32];
        let features = BorrowedTensor::from_f32(feature_data, &[1, 80, INPUT_FRAMES])
            .context("build CoreML feature tensor")?;
        let length =
            BorrowedTensor::from_i32(&length_data, &[1]).context("build CoreML length tensor")?;
        let inputs: [(&str, &dyn AsMultiArray); 2] = [
            ("processed_signal", &features),
            ("processed_signal_length", &length),
        ];
        let prediction = self.model.predict(&inputs).context("CoreML prediction")?;
        let (logits, shape) = prediction.get_f32("logits").context("read CoreML logits")?;
        anyhow::ensure!(
            shape.len() == 3 && shape[0] == 1 && shape[2] == VOCAB_SIZE,
            "CoreML logits shape is {shape:?}, expected [1, T, {VOCAB_SIZE}]"
        );
        let (lengths, _) = prediction
            .get_i32("encoded_lengths")
            .context("read CoreML encoded_lengths")?;
        let t_dim = shape[1];
        let valid = lengths.first().copied().unwrap_or(t_dim as i32).max(0) as usize;
        let valid = valid.min(t_dim);
        let used = valid * VOCAB_SIZE;
        let arr = Array2::from_shape_vec((valid, VOCAB_SIZE), logits[..used].to_vec())?;
        Ok((arr, valid))
    }
}

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
fn validate_coreml_contract(model: &coreml_native::Model) -> Result<()> {
    let inputs = model.inputs();
    let outputs = model.outputs();
    let signal = find_coreml_feature(&inputs, "processed_signal")
        .context("missing CoreML processed_signal")?;
    anyhow::ensure!(
        signal.shape() == Some(&[1, 80, INPUT_FRAMES][..]),
        "CoreML processed_signal shape is {:?}, expected [1, 80, {INPUT_FRAMES}]",
        signal.shape()
    );
    find_coreml_feature(&inputs, "processed_signal_length")
        .context("missing CoreML processed_signal_length")?;
    let logits = find_coreml_feature(&outputs, "logits").context("missing CoreML logits")?;
    anyhow::ensure!(
        logits
            .shape()
            .is_some_and(|shape| shape.len() == 3 && shape[0] == 1 && shape[2] == VOCAB_SIZE),
        "CoreML logits shape is {:?}, expected [1, T, {VOCAB_SIZE}]",
        logits.shape()
    );
    find_coreml_feature(&outputs, "encoded_lengths").context("missing CoreML encoded_lengths")?;
    Ok(())
}

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
fn find_coreml_feature<'a>(
    features: &'a [coreml_native::FeatureDescription],
    name: &str,
) -> Option<&'a coreml_native::FeatureDescription> {
    features.iter().find(|feature| feature.name() == name)
}
