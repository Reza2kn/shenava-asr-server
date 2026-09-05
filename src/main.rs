//! shenava-asr-server — fully-Rust Shenava ASR HTTP server.
//!
//! Endpoints:
//!   POST /transcribe     multipart `file` = wav, optional `hotwords` = newline list
//!   GET  /health         -> `{ok, backend}`
//!
//! One-command bring-up: `./run.sh` downloads the model + tokens from Hugging Face,
//! builds, and starts the server.

mod decode;
mod fbank;
mod model;
#[cfg(feature = "native-diarization")]
mod nemotron;
#[cfg(feature = "native-streaming")]
mod streaming;

use std::{
    fs,
    process::Command,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use clap::Parser;
use ndarray::Array3;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "shenava-asr-server",
    about = "Fully-Rust Shenava ASR server",
    version
)]
struct Args {
    /// Path to the Koochik model: tract ONNX or CoreML .mlpackage/.mlmodelc.
    #[arg(long, default_value = "models/model.onnx")]
    model: String,

    /// Optional cache-aware Persian Koochik CTC streaming ONNX graph.
    #[arg(long)]
    streaming_model: Option<String>,

    /// Token file for the streaming graph (text or its JSON token manifest).
    #[arg(long)]
    streaming_tokens: Option<String>,

    /// Path to tokens.txt (1025 BPE tokens).
    #[arg(long, default_value = "models/tokens.txt")]
    tokens: String,

    /// Path to mel filterbank JSON (80x257).
    #[arg(long, default_value = "assets/mel_filters.json")]
    mel: String,

    /// Path to hotwords file (one per line). Disables hotword boost if omitted.
    #[arg(long)]
    hotwords: Option<String>,

    /// Hotword boost weight.
    #[arg(long, default_value_t = 2.5)]
    hotword_weight: f32,

    /// Beam width for hotword decode.
    #[arg(long, default_value_t = 80)]
    beam: usize,

    /// Compute backend: cpu, gpu-or-cpu (CUDA/Metal auto), or CoreML.
    #[arg(long, value_enum, default_value_t = model::Backend::Cpu)]
    backend: model::Backend,

    /// Listen address.
    #[arg(long, default_value = "0.0.0.0:3000")]
    addr: String,

    /// Optional diarization worker executable. The worker receives
    /// --model, --audio, --backend, and --preset and must print one JSON
    /// response to stdout. Disabled when omitted.
    #[arg(long)]
    diarizer_worker: Option<String>,

    /// Default diarization model selector passed to the worker.
    #[arg(long, default_value = "sortformer")]
    diarizer_model: String,

    /// Diarization execution backend passed to the worker: auto, cpu, or cuda.
    #[arg(long, default_value = "auto")]
    diarizer_backend: String,

    /// Diarization streaming preset passed to the worker.
    #[arg(long, default_value = "very_high_latency")]
    diarizer_preset: String,

    /// Fixed-shape exported Nemotron graph for the native Rust diarizer.
    /// When present, this route is used for `--model nemotron3` without Python
    /// or a native C++ process.
    #[arg(long)]
    diarizer_nemotron_model: Option<String>,

    /// Native Nemotron backend: CPU or Tract's CUDA/CPU auto runtime.
    #[cfg(feature = "native-diarization")]
    #[arg(long, value_enum, default_value_t = model::Backend::GpuOrCpu)]
    diarizer_native_backend: model::Backend,

    /// Native streaming backend: CPU or Tract's CUDA/CPU auto runtime.
    #[cfg(feature = "native-streaming")]
    #[arg(long, value_enum, default_value_t = model::Backend::GpuOrCpu)]
    streaming_backend: model::Backend,
}

#[derive(Clone)]
struct DiarizationConfig {
    worker: String,
    model: String,
    backend: String,
    preset: String,
}

struct App {
    fbank: fbank::Fbank,
    model: model::InferenceModel,
    labels: Vec<String>,
    blank_id: usize,
    hotwords: Vec<String>,
    hotword_weight: f32,
    beam: usize,
    diarization: Option<DiarizationConfig>,
    #[cfg(feature = "native-diarization")]
    nemotron: Option<Arc<nemotron::Model>>,
    #[cfg(feature = "native-streaming")]
    streaming: Option<Arc<streaming::Model>>,
    #[cfg(feature = "native-streaming")]
    streaming_labels: Option<Vec<String>>,
    #[cfg(feature = "native-streaming")]
    streaming_blank_id: Option<usize>,
}

#[derive(Serialize)]
struct TranscribeResp {
    text: String,
    greedy: String,
    elapsed_ms: u64,
    backend: &'static str,
    decoder: &'static str,
    version: &'static str,
    decoder_revision: &'static str,
    mode: &'static str,
    /// Whether this request used the optional native diarization pipeline.
    diarization: bool,
    /// Ordered, speaker-attributed transcript spans. `None` keeps ordinary
    /// ASR responses compact and compatible with existing clients.
    segments: Option<Vec<TranscriptSegment>>,
}

#[derive(Serialize)]
struct TranscriptSegment {
    start_ms: u64,
    end_ms: u64,
    speaker_id: u32,
    speaker: String,
    text: String,
    greedy: String,
}

#[derive(Serialize)]
struct HealthResp {
    ok: bool,
    backend: &'static str,
    version: &'static str,
    decoder_revision: &'static str,
}

#[derive(Serialize, serde::Deserialize)]
struct DiarizeSegment {
    start_ms: u64,
    end_ms: u64,
    speaker_id: u32,
    speaker: String,
}

#[derive(serde::Deserialize)]
struct DiarizeWorkerResp {
    model: String,
    backend: String,
    frame_ms: u32,
    max_speakers: u32,
    segments: Vec<DiarizeSegment>,
}

#[derive(Serialize)]
struct DiarizeResp {
    model: String,
    backend: String,
    frame_ms: u32,
    max_speakers: u32,
    segments: Vec<DiarizeSegment>,
    elapsed_ms: u64,
}

static DIARIZE_REQUEST_ID: AtomicU64 = AtomicU64::new(0);

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let (labels, blank_id) = decode::load_labels(&args.tokens)?;
    log::info!("loaded {} labels (blank={blank_id})", labels.len());
    log::info!("decoder revision: {}", decode::DECODER_REVISION);

    let hotwords = if let Some(p) = &args.hotwords {
        let hw = decode::load_hotwords(p)?;
        log::info!(
            "loaded {} hotwords (weight={}, beam={})",
            hw.len(),
            args.hotword_weight,
            args.beam
        );
        hw
    } else {
        log::warn!("no hotwords file; decoding greedy-only");
        vec![]
    };

    let fb = fbank::Fbank::load(Some(&args.mel))?;
    log::info!("fbank ready (mel 80)");

    let mm = model::InferenceModel::load(&args.model, args.backend)?;
    log::info!("{} model loaded: {}", mm.name(), args.model);

    let diarization = args.diarizer_worker.as_ref().map(|worker| {
        log::info!(
            "diarization worker enabled: {} (model={}, backend={}, preset={})",
            worker,
            args.diarizer_model,
            args.diarizer_backend,
            args.diarizer_preset
        );
        DiarizationConfig {
            worker: worker.clone(),
            model: args.diarizer_model.clone(),
            backend: args.diarizer_backend.clone(),
            preset: args.diarizer_preset.clone(),
        }
    });

    #[cfg(feature = "native-diarization")]
    let nemotron = if let Some(path) = args.diarizer_nemotron_model.as_ref() {
        let loaded = nemotron::Model::load(path, args.diarizer_native_backend)?;
        log::info!("native Nemotron diarizer loaded: {}", path);
        Some(Arc::new(loaded))
    } else {
        None
    };

    #[cfg(not(feature = "native-diarization"))]
    if args.diarizer_nemotron_model.is_some() {
        anyhow::bail!("--diarizer-nemotron-model requires --features native-diarization");
    }

    #[cfg(feature = "native-streaming")]
    let streaming = if let Some(path) = args.streaming_model.as_ref() {
        let loaded = streaming::Model::load(path, args.streaming_backend)?;
        log::info!("native Koochik streaming model loaded: {}", path);
        Some(Arc::new(loaded))
    } else {
        None
    };

    #[cfg(feature = "native-streaming")]
    let (streaming_labels, streaming_blank_id) = if args.streaming_model.is_some() {
        let path = args.streaming_tokens.as_ref().unwrap_or(&args.tokens);
        let (labels, blank_id) = decode::load_labels(path)?;
        log::info!(
            "loaded {} streaming labels (blank={blank_id})",
            labels.len()
        );
        (Some(labels), Some(blank_id))
    } else {
        (None, None)
    };

    #[cfg(not(feature = "native-streaming"))]
    if args.streaming_model.is_some() {
        anyhow::bail!("--streaming-model requires --features native-streaming");
    }

    let app = Arc::new(App {
        fbank: fb,
        model: mm,
        labels,
        blank_id,
        hotwords,
        hotword_weight: args.hotword_weight,
        beam: args.beam,
        diarization,
        #[cfg(feature = "native-diarization")]
        nemotron,
        #[cfg(feature = "native-streaming")]
        streaming,
        #[cfg(feature = "native-streaming")]
        streaming_labels,
        #[cfg(feature = "native-streaming")]
        streaming_blank_id,
    });

    let router = Router::new()
        .route("/health", get(health))
        .route("/transcribe", post(transcribe))
        .route("/diarize", post(diarize))
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .with_state(app);

    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    log::info!("shenava-asr-server listening on {}", args.addr);
    axum::serve(listener, router).await?;
    Ok(())
}

async fn diarize(
    State(app): State<Arc<App>>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let config = app.diarization.clone();
    #[cfg(feature = "native-diarization")]
    let native_nemotron = app.nemotron.clone();
    if config.is_none() && {
        #[cfg(feature = "native-diarization")]
        {
            native_nemotron.is_none()
        }
        #[cfg(not(feature = "native-diarization"))]
        {
            true
        }
    } {
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            "diarization is disabled; start with --diarizer-worker or --diarizer-nemotron-model"
                .into(),
        ));
    }

    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut requested_model: Option<String> = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        match field.name() {
            Some("file") => {
                audio_bytes = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
                        .to_vec(),
                );
            }
            Some("model") => {
                requested_model = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?,
                );
            }
            _ => {}
        }
    }
    let Some(bytes) = audio_bytes else {
        return Err((
            StatusCode::BAD_REQUEST,
            "missing multipart field `file`".into(),
        ));
    };
    let model = requested_model
        .or_else(|| config.as_ref().map(|value| value.model.clone()))
        .unwrap_or_else(|| "nemotron3".to_string());
    if !valid_worker_selector(&model) {
        return Err((
            StatusCode::BAD_REQUEST,
            "invalid diarization model selector".into(),
        ));
    }

    let started = std::time::Instant::now();
    let result = tokio::task::spawn_blocking(move || {
        #[cfg(feature = "native-diarization")]
        if model_is_nemotron(&model) {
            if let Some(native) = native_nemotron {
                return native_diarize_bytes(&native, &bytes);
            }
        }
        let config =
            config.ok_or_else(|| anyhow::anyhow!("native Nemotron diarizer is not configured"))?;
        diarize_bytes(&config, &bytes, &model)
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("diarization worker failed: {e}"),
        )
    })?;
    match result {
        Ok(mut worker) => {
            worker
                .segments
                .sort_by_key(|segment| (segment.start_ms, segment.end_ms, segment.speaker_id));
            Ok((
                StatusCode::OK,
                Json(DiarizeResp {
                    model: worker.model,
                    backend: worker.backend,
                    frame_ms: worker.frame_ms,
                    max_speakers: worker.max_speakers,
                    segments: worker.segments,
                    elapsed_ms: started.elapsed().as_millis() as u64,
                }),
            ))
        }
        Err(e) => Err((
            StatusCode::BAD_GATEWAY,
            format!("diarization worker error: {e:#}"),
        )),
    }
}

fn valid_worker_selector(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 80
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

#[cfg(feature = "native-diarization")]
fn model_is_nemotron(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "nemotron" | "nemotron3" | "nemotron-3-diarization-preview"
    )
}

#[cfg(feature = "native-diarization")]
fn native_diarize_bytes(model: &nemotron::Model, bytes: &[u8]) -> Result<DiarizeWorkerResp> {
    let segments = model.diarize_wav(bytes)?;
    Ok(DiarizeWorkerResp {
        model: "nemotron3".into(),
        backend: model.backend().into(),
        frame_ms: nemotron::FRAME_MS as u32,
        max_speakers: nemotron::SPEAKERS as u32,
        segments: segments
            .into_iter()
            .map(|segment| DiarizeSegment {
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                speaker_id: segment.speaker_id,
                speaker: format!("speaker_{}", segment.speaker_id),
            })
            .collect(),
    })
}

fn diarize_bytes(
    config: &DiarizationConfig,
    bytes: &[u8],
    model: &str,
) -> Result<DiarizeWorkerResp> {
    let request_id = DIARIZE_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let audio_path = std::env::temp_dir().join(format!(
        "shenava-diar-{}-{}-{}.wav",
        std::process::id(),
        stamp,
        request_id
    ));
    fs::write(&audio_path, bytes)
        .with_context(|| format!("write diarization audio {}", audio_path.display()))?;

    let output = Command::new(&config.worker)
        .args([
            "--model",
            model,
            "--audio",
            audio_path
                .to_str()
                .context("temporary audio path is not UTF-8")?,
            "--backend",
            &config.backend,
            "--preset",
            &config.preset,
        ])
        .output()
        .with_context(|| format!("spawn diarization worker {}", config.worker));
    let _ = fs::remove_file(&audio_path);
    let output = output?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("worker exited with {}: {}", output.status, stderr.trim());
    }
    let stdout =
        String::from_utf8(output.stdout).context("diarization worker stdout is not UTF-8")?;
    // Some NeMo versions attach an informational handler to stdout. The
    // worker contract is still one JSON object, so select the last JSON line
    // rather than making the Rust bridge depend on NeMo's logger stream.
    let payload = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .context("diarization worker returned no JSON object")?;
    serde_json::from_str(payload.trim()).context("invalid diarization worker JSON")
}

async fn health(State(app): State<Arc<App>>) -> Json<HealthResp> {
    Json(HealthResp {
        ok: true,
        backend: app.model.name(),
        version: env!("CARGO_PKG_VERSION"),
        decoder_revision: decode::DECODER_REVISION,
    })
}

async fn transcribe(
    State(app): State<Arc<App>>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut request_hotwords = Vec::new();
    let mut requested_diarization = false;
    let mut requested_streaming = false;
    let mut mode_was_explicit = false;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        match field.name() {
            Some("file") => {
                audio_bytes = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
                        .to_vec(),
                );
            }
            Some("hotwords") => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
                request_hotwords.extend(decode::parse_hotwords(&text));
            }
            Some("diarization") => {
                let value = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
                requested_diarization = parse_bool_field(&value).ok_or_else(|| {
                    (
                        StatusCode::BAD_REQUEST,
                        "`diarization` must be true/false, 1/0, yes/no, or on/off".into(),
                    )
                })?;
            }
            Some("mode") => {
                mode_was_explicit = true;
                let value = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
                requested_streaming = parse_mode(&value).ok_or_else(|| {
                    (
                        StatusCode::BAD_REQUEST,
                        "`mode` must be offline or streaming".into(),
                    )
                })?;
            }
            _ => {}
        }
    }
    let Some(bytes) = audio_bytes else {
        return Err((
            StatusCode::BAD_REQUEST,
            "missing multipart field `file`".into(),
        ));
    };
    #[cfg(feature = "native-streaming")]
    if let Some(_streaming) = app.streaming.as_ref() {
        let (signal, sample_rate) = fbank::read_wav_bytes(&bytes)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("{e:#}")))?;
        if should_auto_stream(
            signal.len(),
            sample_rate,
            true,
            mode_was_explicit,
            requested_diarization,
        ) {
            log::info!(
                "audio is longer than the offline model window; using native streaming fallback"
            );
            requested_streaming = true;
        }
    }
    if requested_diarization && requested_streaming {
        return Err((
            StatusCode::BAD_REQUEST,
            "`diarization=true` currently uses the offline Koochik segment transcriber; choose mode=offline"
                .into(),
        ));
    }
    #[cfg(feature = "native-diarization")]
    let native_nemotron = app.nemotron.clone();
    #[cfg(feature = "native-streaming")]
    let native_streaming = app.streaming.clone();
    if requested_diarization {
        #[cfg(feature = "native-diarization")]
        if native_nemotron.is_none() {
            return Err((
                StatusCode::NOT_IMPLEMENTED,
                "diarization was requested but no native Nemotron model is configured; start with --diarizer-nemotron-model"
                    .into(),
            ));
        }
        #[cfg(not(feature = "native-diarization"))]
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            "diarization was requested but this binary was built without native-diarization".into(),
        ));
    }
    if requested_streaming {
        #[cfg(feature = "native-streaming")]
        if native_streaming.is_none() {
            return Err((
                StatusCode::NOT_IMPLEMENTED,
                "streaming mode was requested but no streaming model is configured; start with --streaming-model"
                    .into(),
            ));
        }
        #[cfg(not(feature = "native-streaming"))]
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            "streaming mode was requested but this binary was built without native-streaming"
                .into(),
        ));
    }
    let response_backend = if requested_streaming {
        #[cfg(feature = "native-streaming")]
        {
            app.streaming
                .as_ref()
                .map(|model| model.backend())
                .unwrap_or("streaming")
        }
        #[cfg(not(feature = "native-streaming"))]
        {
            "streaming"
        }
    } else {
        app.model.name()
    };
    let start = std::time::Instant::now();
    let worker_app = app.clone();
    let result = tokio::task::spawn_blocking(
        move || -> Result<(String, String, bool, Option<Vec<TranscriptSegment>>)> {
            if requested_diarization {
                #[cfg(feature = "native-diarization")]
                {
                    let native = native_nemotron
                        .as_ref()
                        .expect("native model checked before spawn_blocking");
                    return transcribe_diarized(&worker_app, native, &bytes, &request_hotwords);
                }
                #[cfg(not(feature = "native-diarization"))]
                unreachable!("native diarization request rejected above");
            }
            if requested_streaming {
                #[cfg(feature = "native-streaming")]
                {
                    let streaming = native_streaming
                        .as_ref()
                        .expect("streaming model checked before spawn_blocking");
                    let (text, greedy, used_hotbeam) =
                        transcribe_streaming(&worker_app, streaming, &bytes, &request_hotwords)?;
                    return Ok((text, greedy, used_hotbeam, None));
                }
                #[cfg(not(feature = "native-streaming"))]
                unreachable!("native streaming request rejected above");
            }
            let (text, greedy, used_hotbeam) =
                transcribe_bytes(worker_app, &bytes, &request_hotwords)?;
            Ok((text, greedy, used_hotbeam, None))
        },
    )
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("inference worker failed: {e}"),
        )
    })?;
    let elapsed_ms = start.elapsed().as_millis() as u64;
    match result {
        Ok((text, greedy, used_hotbeam, segments)) => Ok((
            StatusCode::OK,
            Json(TranscribeResp {
                text,
                greedy,
                elapsed_ms,
                backend: response_backend,
                decoder: if used_hotbeam { "hotbeam" } else { "greedy" },
                version: env!("CARGO_PKG_VERSION"),
                decoder_revision: decode::DECODER_REVISION,
                mode: if requested_streaming {
                    "streaming"
                } else {
                    "offline"
                },
                diarization: segments.is_some(),
                segments,
            }),
        )),
        Err(e) => Err((StatusCode::BAD_REQUEST, format!("{e:#}"))),
    }
}

fn parse_bool_field(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_mode(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "offline" => Some(false),
        "streaming" | "stream" => Some(true),
        _ => None,
    }
}
const CHUNK_SECONDS: usize = 10;

fn should_auto_stream(
    sample_count: usize,
    sample_rate: u32,
    streaming_available: bool,
    mode_was_explicit: bool,
    diarization_requested: bool,
) -> bool {
    streaming_available
        && !mode_was_explicit
        && !diarization_requested
        && sample_rate > 0
        && sample_count > sample_rate as usize * CHUNK_SECONDS * 2
}

fn transcribe_bytes(
    app: Arc<App>,
    bytes: &[u8],
    request_hotwords: &[String],
) -> Result<(String, String, bool)> {
    let (sig, sr) = fbank::read_wav_bytes(bytes)?;
    log::debug!("wav {sr} Hz, {} samples", sig.len());

    transcribe_signal(&app, &sig, sr, request_hotwords)
}

fn transcribe_signal(
    app: &App,
    sig: &[f32],
    sr: u32,
    request_hotwords: &[String],
) -> Result<(String, String, bool)> {
    let chunk_len = sr as usize * CHUNK_SECONDS;
    let chunks: Vec<&[f32]> = sig.chunks(chunk_len).collect();
    log::debug!(
        "split {} samples into {} chunk(s) of up to {}s",
        sig.len(),
        chunks.len(),
        CHUNK_SECONDS
    );

    let mut texts = Vec::with_capacity(chunks.len());
    let mut greedies = Vec::with_capacity(chunks.len());
    let mut used_hotbeam = false;
    for (i, chunk) in chunks.iter().enumerate() {
        let (text, greedy, chunk_used_hotbeam) =
            transcribe_single_signal(app, chunk, sr, request_hotwords).with_context(|| {
                format!("transcribe offline chunk {i} (up to {CHUNK_SECONDS}s)")
            })?;
        texts.push(text);
        greedies.push(greedy);
        used_hotbeam |= chunk_used_hotbeam;
    }

    Ok((texts.join(" "), greedies.join(" "), used_hotbeam))
}

fn transcribe_single_signal(
    app: &App,
    sig: &[f32],
    sr: u32,
    request_hotwords: &[String],
) -> Result<(String, String, bool)> {
    let (feat, nf) = app.fbank.process(&sig, sr)?;
    anyhow::ensure!(
        nf <= model::INPUT_FRAMES,
        "audio is too long for the fixed 2005-frame model window (about 20 seconds)"
    );
    let fixed: Array3<f32> = app.fbank.to_fixed(&feat, nf);
    log::debug!("fbank {} frames -> fixed 2005", nf);

    let (log_probs, valid) = app.model.run(&fixed, nf)?;
    log::debug!("{} log_probs [{valid}, 1025]", app.model.name());

    let greedy = decode::greedy(&log_probs, &app.labels, app.blank_id);
    let mut hotwords = app.hotwords.clone();
    hotwords.extend_from_slice(request_hotwords);
    let used_hotbeam = !hotwords.is_empty();
    let text = if !used_hotbeam {
        greedy.clone()
    } else {
        decode::decode_hotword(
            &log_probs,
            &app.labels,
            &hotwords,
            app.hotword_weight,
            app.beam,
        )
    };
    Ok((text, greedy, used_hotbeam))
}

#[cfg(feature = "native-streaming")]
fn transcribe_streaming(
    app: &App,
    streaming: &streaming::Model,
    bytes: &[u8],
    request_hotwords: &[String],
) -> Result<(String, String, bool)> {
    let (sig, sr) = fbank::read_wav_bytes(bytes)?;
    let (feat, nf) = app.fbank.process(&sig, sr)?;
    let log_probs = streaming.run_features(&feat, nf)?;
    let labels = app
        .streaming_labels
        .as_ref()
        .context("streaming token labels are not configured")?;
    let blank_id = app
        .streaming_blank_id
        .context("streaming blank id is not configured")?;
    let greedy = decode::greedy(&log_probs, labels, blank_id);
    let mut hotwords = app.hotwords.clone();
    hotwords.extend_from_slice(request_hotwords);
    let used_hotbeam = !hotwords.is_empty();
    let text = if !used_hotbeam {
        greedy.clone()
    } else {
        decode::decode_hotword(&log_probs, labels, &hotwords, app.hotword_weight, app.beam)
    };
    Ok((text, greedy, used_hotbeam))
}

#[cfg(feature = "native-diarization")]
fn transcribe_diarized(
    app: &App,
    native: &nemotron::Model,
    bytes: &[u8],
    request_hotwords: &[String],
) -> Result<(String, String, bool, Option<Vec<TranscriptSegment>>)> {
    let diarization = native_diarize_bytes(native, bytes)?;
    let (sig, sr) = fbank::read_wav_bytes(bytes)?;
    let mut segments = Vec::with_capacity(diarization.segments.len());
    let mut ordered_text = Vec::with_capacity(diarization.segments.len());
    let mut ordered_greedy = Vec::with_capacity(diarization.segments.len());

    for segment in diarization.segments {
        let start =
            ((segment.start_ms as u128 * sr as u128) / 1000).min(sig.len() as u128) as usize;
        let end = ((segment.end_ms as u128 * sr as u128) / 1000).min(sig.len() as u128) as usize;
        let (text, greedy) = if end > start {
            let (text, greedy, _) = transcribe_signal(app, &sig[start..end], sr, request_hotwords)
                .with_context(|| {
                    format!(
                        "transcribe {} from {}ms to {}ms",
                        segment.speaker, segment.start_ms, segment.end_ms
                    )
                })?;
            (text, greedy)
        } else {
            (String::new(), String::new())
        };

        if !text.is_empty() {
            ordered_text.push(format!("{}: {}", segment.speaker, text));
        }
        if !greedy.is_empty() {
            ordered_greedy.push(format!("{}: {}", segment.speaker, greedy));
        }
        segments.push(TranscriptSegment {
            start_ms: segment.start_ms,
            end_ms: segment.end_ms,
            speaker_id: segment.speaker_id,
            speaker: segment.speaker,
            text,
            greedy,
        });
    }

    Ok((
        ordered_text.join("\n"),
        ordered_greedy.join("\n"),
        !request_hotwords.is_empty() || !app.hotwords.is_empty(),
        Some(segments),
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        parse_bool_field, parse_mode, should_auto_stream, valid_worker_selector, DiarizeWorkerResp,
    };

    #[test]
    fn boolean_request_field_accepts_common_multipart_values() {
        for value in ["true", "TRUE", "1", "yes", " on "] {
            assert_eq!(parse_bool_field(value), Some(true));
        }
        for value in ["false", "FALSE", "0", "no", " off "] {
            assert_eq!(parse_bool_field(value), Some(false));
        }
        assert_eq!(parse_bool_field("maybe"), None);
    }

    #[test]
    fn mode_field_defaults_to_offline_semantics() {
        assert_eq!(parse_mode("offline"), Some(false));
        assert_eq!(parse_mode("streaming"), Some(true));
        assert_eq!(parse_mode("stream"), Some(true));
        assert_eq!(parse_mode("batch"), None);
    }

    #[test]
    fn long_audio_auto_selects_streaming_only_when_mode_is_omitted() {
        assert!(should_auto_stream(16000 * 21, 16000, true, false, false));
        assert!(!should_auto_stream(16000 * 21, 16000, true, true, false));
        assert!(!should_auto_stream(16000 * 21, 16000, true, false, true));
        assert!(!should_auto_stream(16000 * 21, 16000, false, false, false));
        assert!(!should_auto_stream(16000 * 20, 16000, true, false, false));
    }

    #[test]
    fn worker_selector_rejects_path_and_empty_values() {
        assert!(valid_worker_selector("sortformer"));
        assert!(valid_worker_selector("nemotron3"));
        assert!(!valid_worker_selector(""));
        assert!(!valid_worker_selector("../model"));
        assert!(!valid_worker_selector("nvidia/model"));
    }

    #[test]
    fn worker_response_contract_parses_segments() {
        let response: DiarizeWorkerResp = serde_json::from_str(
            r#"{
                "model":"sortformer",
                "backend":"cpu",
                "frame_ms":80,
                "max_speakers":4,
                "segments":[{"start_ms":1040,"end_ms":2240,"speaker_id":1,"speaker":"speaker_1"}]
            }"#,
        )
        .expect("worker response should parse");
        assert_eq!(response.model, "sortformer");
        assert_eq!(response.frame_ms, 80);
        assert_eq!(response.segments[0].speaker_id, 1);
        assert_eq!(response.segments[0].end_ms, 2240);
    }
}
