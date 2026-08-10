//! Hotword-boosted CTC beam decode via `shenava-ctc-beam`.
//!
//! Labels are built from the BPE token file: `<...>` special tokens are remapped to PUA
//! `\u{E000}+i` so they survive the beam (then stripped), and the blank token (last entry,
//! `<blk>`) becomes the empty string which the decoder treats as the CTC blank.

use anyhow::Result;
use ndarray::Array2;
use shenava_ctc_beam::{CtcBeamDecoder, Hotwords};

pub const BPE: char = '\u{2581}'; // ▁

/// Load token labels from a `tokens.txt` (one `token id` per line, id ascending from 0).
/// Returns `(labels, blank_id)`.
pub fn load_labels(tokens_path: &str) -> Result<(Vec<String>, usize)> {
    let txt = std::fs::read_to_string(tokens_path)?;
    let mut toks: Vec<String> = Vec::new();
    for line in txt.lines() {
        let t = line.split_whitespace().next().unwrap_or_default();
        toks.push(t.to_string());
    }
    if toks.is_empty() {
        anyhow::bail!("empty tokens.txt");
    }
    // blank is the last token (`<blk>`), mapped to the empty label for the decoder.
    let blank_id = toks.len() - 1;
    let mut labels: Vec<String> = Vec::with_capacity(toks.len());
    for (i, t) in toks.iter().enumerate() {
        if i == blank_id {
            labels.push(String::new());
        } else if t.starts_with('<') && t.ends_with('>') {
            labels.push(char::from_u32(0xE000 + i as u32).unwrap().to_string());
        } else {
            labels.push(t.clone());
        }
    }
    Ok((labels, blank_id))
}

/// Greedy argmax decode (no hotwords) — useful as a fast baseline.
pub fn greedy(log_probs: &Array2<f32>, labels: &[String], blank_id: usize) -> String {
    let mut out = String::new();
    let mut prev: i64 = -1;
    for t in 0..log_probs.shape()[0] {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for v in 0..log_probs.shape()[1] {
            let x = log_probs[[t, v]];
            if x > best_v {
                best_v = x;
                best = v;
            }
        }
        if best == blank_id || best as i64 == prev {
            continue;
        }
        let s = &labels[best];
        if s.is_empty() {
            continue;
        }
        let s = s.trim_start_matches(BPE).trim_end_matches(BPE).to_string();
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&s);
        prev = best as i64;
    }
    out
}

/// Run hotword-boosted beam decode over `log_probs` (T x V).
pub fn decode_hotword(
    log_probs: &Array2<f32>,
    labels: &[String],
    hotwords: &[String],
    hotword_weight: f32,
    beam_width: usize,
) -> String {
    let dec = CtcBeamDecoder::new(labels.to_vec());
    let hw = Hotwords::new(hotwords.to_vec(), hotword_weight);
    let rows: Vec<Vec<f32>> = (0..log_probs.shape()[0])
        .map(|t| (0..log_probs.shape()[1]).map(|v| log_probs[[t, v]]).collect())
        .collect();
    let text = dec.decode(&rows, &hw, beam_width, -5.0, -10.0);
    text.chars()
        .filter(|c| !('\u{E000}'..='\u{E0FF}').contains(c))
        .collect::<String>()
}

/// Load a hotword list (one word/phrase per line; blank lines ignored).
pub fn load_hotwords(path: &str) -> Result<Vec<String>> {
    let txt = std::fs::read_to_string(path)?;
    Ok(txt
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect())
}
