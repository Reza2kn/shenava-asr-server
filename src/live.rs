//! Browser microphone and bounded, connection-local streaming sessions.
use crate::App;
use axum::{
    extract::{State, WebSocketUpgrade},
    http::{HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
};
use std::sync::Arc;

pub async fn page() -> Html<&'static str> {
    Html(include_str!("../web/mic.html"))
}
pub async fn worklet() -> impl IntoResponse {
    (
        [
            ("content-type", "application/javascript"),
            ("cache-control", "no-cache"),
        ],
        include_str!("../web/mic-worklet.js"),
    )
}

pub async fn script() -> impl IntoResponse {
    (
        [
            ("content-type", "application/javascript"),
            ("cache-control", "no-cache"),
        ],
        include_str!("../web/mic.js"),
    )
}
pub async fn config(State(app): State<Arc<App>>) -> axum::Json<serde_json::Value> {
    #[cfg(feature = "native-streaming")]
    let available = app.streaming.is_some();
    #[cfg(not(feature = "native-streaming"))]
    let available = {
        let _ = app;
        false
    };
    axum::Json(
        serde_json::json!({"available": available, "message": "Live microphone needs the streaming model. Start the server with ./run.sh (streaming enabled), or build with --features native-streaming and configure --streaming-model and --streaming-tokens. See docs/MICROPHONE.md."}),
    )
}

pub async fn upgrade(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    // Browsers may only open this connection from the server's own page.
    if let Some(origin) = headers.get("origin") {
        let origin = origin.to_str().unwrap_or("");
        let authority = origin
            .strip_prefix("https://")
            .or_else(|| origin.strip_prefix("http://"));
        if authority.is_none() || authority != headers.get("host").and_then(|h| h.to_str().ok()) {
            return (
                StatusCode::FORBIDDEN,
                "microphone page and WebSocket must have the same origin",
            )
                .into_response();
        }
    }
    #[cfg(feature = "native-streaming")]
    {
        if app.streaming.is_none() {
            return (StatusCode::SERVICE_UNAVAILABLE, "Live microphone requires --streaming-model and --streaming-tokens. Start with ./run.sh.").into_response();
        }
        // Bound per-connection caches and simultaneous inference work.
        static SESSIONS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(2);
        let permit = match SESSIONS.try_acquire() {
            Ok(permit) => permit,
            Err(_) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Both microphone sessions are busy",
                )
                    .into_response()
            }
        };
        ws.max_message_size(6400)
            .max_frame_size(6400)
            .on_upgrade(move |socket| async move {
                let _permit = permit;
                session(socket, app).await;
            })
    }
    #[cfg(not(feature = "native-streaming"))]
    {
        let _ = (app, ws);
        (StatusCode::SERVICE_UNAVAILABLE, "Live microphone requires a build with --features native-streaming and a configured streaming model. Start with ./run.sh.").into_response()
    }
}

#[cfg(feature = "native-streaming")]
struct Session {
    features: crate::fbank::LiveFeatures,
    cache: crate::streaming::CacheState,
    decoder: crate::decode::GreedyStream,
    samples: usize,
}

#[cfg(feature = "native-streaming")]
impl Session {
    fn new() -> Self {
        Self {
            features: Default::default(),
            cache: crate::streaming::CacheState::new(),
            decoder: Default::default(),
            samples: 0,
        }
    }
    fn process(
        &mut self,
        app: &App,
        bytes: &[u8],
        finish: bool,
    ) -> anyhow::Result<serde_json::Value> {
        anyhow::ensure!(
            bytes.len() % 2 == 0,
            "PCM16 packets must have an even byte count"
        );
        self.samples += bytes.len() / 2;
        anyhow::ensure!(
            self.samples <= 16000 * 60 * 30,
            "30-minute session limit reached; start a new recording"
        );
        let pcm: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();
        self.features.push(&pcm);
        let mut changed = false;
        while let Some((features, valid)) = self.features.next_chunk(&app.fbank, finish)? {
            let probs =
                app.streaming
                    .as_ref()
                    .unwrap()
                    .run_chunk(features, valid, &mut self.cache)?;
            self.decoder.push(
                &probs,
                app.streaming_labels.as_ref().unwrap(),
                app.streaming_blank_id.unwrap(),
            );
            changed = true;
        }
        let mut result = serde_json::json!({"type": if finish { "final" } else { "ack" }, "audio_ms": self.samples * 1000 / 16000});
        if changed || finish {
            result["text"] = self.decoder.text().into();
        }
        Ok(result)
    }
}

#[cfg(feature = "native-streaming")]
async fn session(mut socket: axum::extract::ws::WebSocket, app: Arc<App>) {
    use axum::extract::ws::Message;
    use std::time::Duration;
    let ready = serde_json::json!({"type":"ready", "sample_rate":16000, "channels":1, "format":"pcm_s16le", "max_packet_samples":3200, "decoder":"greedy", "backend": app.streaming.as_ref().unwrap().backend()});
    if socket
        .send(Message::Text(ready.to_string().into()))
        .await
        .is_err()
    {
        return;
    }
    let mut state = Session::new();
    loop {
        let message = match tokio::time::timeout(Duration::from_secs(60), socket.recv()).await {
            Ok(Some(Ok(message))) => message,
            _ => break,
        };
        let (bytes, finish) = match message {
            Message::Binary(bytes) => (bytes.to_vec(), false),
            Message::Text(text)
                if serde_json::from_str::<serde_json::Value>(&text)
                    .ok()
                    .and_then(|v| {
                        v.get("type")
                            .and_then(|v| v.as_str())
                            .map(|v| v == "finish")
                    })
                    .unwrap_or(false) =>
            {
                (vec![], true)
            }
            Message::Close(_) => break,
            Message::Ping(_) | Message::Pong(_) => continue,
            _ => {
                let _ = socket.send(Message::Text(r#"{"type":"error","message":"Send binary PCM16 or {\"type\":\"finish\"}"}"#.into())).await;
                break;
            }
        };
        let worker_app = app.clone();
        let output = tokio::task::spawn_blocking(move || {
            let result = state.process(&worker_app, &bytes, finish);
            (state, result)
        })
        .await;
        let response = match output {
            Ok((next, Ok(response))) => {
                state = next;
                response
            }
            Ok((_, Err(error))) => {
                let _ = socket
                    .send(Message::Text(
                        serde_json::json!({"type":"error", "message": error.to_string()})
                            .to_string()
                            .into(),
                    ))
                    .await;
                break;
            }
            Err(error) => {
                log::error!("microphone inference worker failed: {error}");
                let _ = socket
                    .send(Message::Text(
                        r#"{"type":"error","message":"Inference worker failed"}"#.into(),
                    ))
                    .await;
                break;
            }
        };
        if socket
            .send(Message::Text(response.to_string().into()))
            .await
            .is_err()
            || finish
        {
            break;
        }
    }
    let _ = socket.send(Message::Close(None)).await;
}
