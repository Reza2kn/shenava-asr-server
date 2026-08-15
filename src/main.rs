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

use std::sync::Arc;

use anyhow::Result;
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
#[command(name = "shenava-asr-server", about = "Fully-Rust Shenava ASR server")]
struct Args {
    /// Path to the Koochik model: tract ONNX or CoreML .mlpackage/.mlmodelc.
    #[arg(long, default_value = "models/model.onnx")]
    model: String,

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
}

struct App {
    fbank: fbank::Fbank,
    model: model::InferenceModel,
    labels: Vec<String>,
    blank_id: usize,
    hotwords: Vec<String>,
    hotword_weight: f32,
    beam: usize,
}

#[derive(Serialize)]
struct TranscribeResp {
    text: String,
    greedy: String,
    elapsed_ms: u64,
    backend: &'static str,
    decoder: &'static str,
}

#[derive(Serialize)]
struct HealthResp {
    ok: bool,
    backend: &'static str,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let (labels, blank_id) = decode::load_labels(&args.tokens)?;
    log::info!("loaded {} labels (blank={blank_id})", labels.len());

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

    let app = Arc::new(App {
        fbank: fb,
        model: mm,
        labels,
        blank_id,
        hotwords,
        hotword_weight: args.hotword_weight,
        beam: args.beam,
    });

    let router = Router::new()
        .route("/health", get(health))
        .route("/transcribe", post(transcribe))
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .with_state(app);

    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    log::info!("shenava-asr-server listening on {}", args.addr);
    axum::serve(listener, router).await?;
    Ok(())
}

async fn health(State(app): State<Arc<App>>) -> Json<HealthResp> {
    Json(HealthResp {
        ok: true,
        backend: app.model.name(),
    })
}

async fn transcribe(
    State(app): State<Arc<App>>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut request_hotwords = Vec::new();
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
            _ => {}
        }
    }
    let Some(bytes) = audio_bytes else {
        return Err((
            StatusCode::BAD_REQUEST,
            "missing multipart field `file`".into(),
        ));
    };
    let start = std::time::Instant::now();
    let worker_app = app.clone();
    let result = tokio::task::spawn_blocking(move || {
        transcribe_bytes(worker_app, &bytes, &request_hotwords)
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("inference worker failed: {e}"),
        )
    })?;
    let elapsed_ms = start.elapsed().as_millis() as u64;
    match result {
        Ok((text, greedy, used_hotbeam)) => Ok((
            StatusCode::OK,
            Json(TranscribeResp {
                text,
                greedy,
                elapsed_ms,
                backend: app.model.name(),
                decoder: if used_hotbeam { "hotbeam" } else { "greedy" },
            }),
        )),
        Err(e) => Err((StatusCode::BAD_REQUEST, format!("{e:#}"))),
    }
}

fn transcribe_bytes(
    app: Arc<App>,
    bytes: &[u8],
    request_hotwords: &[String],
) -> Result<(String, String, bool)> {
    let (sig, sr) = fbank::read_wav_bytes(bytes)?;
    log::debug!("wav {sr} Hz, {} samples", sig.len());

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
