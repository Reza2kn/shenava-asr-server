//! shenava-asr-server — fully-Rust Shenava ASR HTTP server.
//!
//! Endpoints:
//!   POST /transcribe     (multipart `file` = wav) -> `{text, greedy, hotword, elapsed_ms}`
//!   GET  /health         -> `{ok}`
//!
//! One-command bring-up: `./run.sh` downloads the model + tokens from Hugging Face,
//! builds, and starts the server.

mod decode;
mod fbank;
mod model;

use std::sync::{Arc, Mutex};

use anyhow::Result;
use axum::{
    extract::{Multipart, State},
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
    /// Path to the pre-simplified Koochik ONNX.
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

    /// Compute backend: gpu-or-cpu (auto: CUDA/Metal→CPU) or cpu.
    #[arg(long, value_enum, default_value_t = model::Backend::GpuOrCpu)]
    backend: model::Backend,

    /// Listen address.
    #[arg(long, default_value = "0.0.0.0:3000")]
    addr: String,
}

struct App {
    fbank: Mutex<fbank::Fbank>,
    model: model::KoochikModel,
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
}

#[derive(Serialize)]
struct HealthResp {
    ok: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let (labels, blank_id) = decode::load_labels(&args.tokens)?;
    log::info!("loaded {} labels (blank={blank_id})", labels.len());

    let hotwords = if let Some(p) = &args.hotwords {
        let hw = decode::load_hotwords(p)?;
        log::info!("loaded {} hotwords (weight={}, beam={})", hw.len(), args.hotword_weight, args.beam);
        hw
    } else {
        log::warn!("no hotwords file; decoding greedy-only");
        vec![]
    };

    let fb = fbank::Fbank::load(Some(&args.mel))?;
    log::info!("fbank ready (mel 80)");

    let mm = model::KoochikModel::load(&args.model, args.backend)?;
    log::info!("tract model loaded: {}", args.model);

    let app = Arc::new(App {
        fbank: Mutex::new(fb),
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
        .with_state(app);

    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    log::info!("shenava-asr-server listening on {}", args.addr);
    axum::serve(listener, router).await?;
    Ok(())
}

async fn health() -> Json<HealthResp> {
    Json(HealthResp { ok: true })
}

async fn transcribe(
    State(app): State<Arc<App>>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut audio_bytes: Option<Vec<u8>> = None;
    while let Some(field) = multipart.next_field().await.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))? {
        if field.name() == Some("file") {
            audio_bytes = Some(field.bytes().await.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?.to_vec());
            break;
        }
    }
    let Some(bytes) = audio_bytes else {
        return Err((StatusCode::BAD_REQUEST, "missing multipart field `file`".into()));
    };
    let start = std::time::Instant::now();
    let result = transcribe_bytes(app, &bytes);
    let elapsed_ms = start.elapsed().as_millis() as u64;
    match result {
        Ok((text, greedy)) => Ok((StatusCode::OK, Json(TranscribeResp { text, greedy, elapsed_ms }))),
        Err(e) => Err((StatusCode::BAD_REQUEST, format!("{e:#}"))),
    }
}

fn transcribe_bytes(app: Arc<App>, bytes: &[u8]) -> Result<(String, String)> {
    // write to temp wav
    let tmp = std::env::temp_dir().join(format!("shenava_{}.wav", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    let (sig, sr) = fbank::read_wav(tmp.to_str().unwrap())?;
    std::fs::remove_file(&tmp).ok();
    log::debug!("wav {sr} Hz, {} samples", sig.len());

    let mut fb = app.fbank.lock().unwrap();
    let (feat, nf) = fb.process(&sig, sr)?;
    let fixed: Array3<f32> = fb.to_fixed(&feat, nf);
    log::debug!("fbank {} frames -> fixed 2005", nf);

    let (log_probs, valid) = app.model.run(&fixed, nf)?;
    log::debug!("tract log_probs [{valid}, 1025]");

    let greedy = decode::greedy(&log_probs, &app.labels, app.blank_id);
    let hotword = if app.hotwords.is_empty() {
        greedy.clone()
    } else {
        decode::decode_hotword(
            &log_probs,
            &app.labels,
            &app.hotwords,
            app.hotword_weight,
            app.beam,
        )
    };
    Ok((hotword, greedy))
}
