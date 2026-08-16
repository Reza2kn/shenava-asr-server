//! Hotword-boosted CTC beam decode via `shenava-ctc-beam`.
//!
//! Labels are built from the BPE token file: `<...>` special tokens are remapped to PUA
//! `\u{E000}+i` so they survive the beam (then stripped), and the blank token (last entry,
//! `<blk>`) becomes the empty string which the decoder treats as the CTC blank.

use anyhow::Result;
use ndarray::Array2;
use shenava_ctc_beam::{CtcBeamDecoder, Hotwords};

pub const BPE: char = '\u{2581}'; // ▁
pub const DECODER_REVISION: &str = "sentencepiece-v2";

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
    let mut prev: Option<usize> = None;
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
        if best == blank_id {
            prev = None;
            continue;
        }
        if prev == Some(best) {
            continue;
        }
        let s = &labels[best];
        if s.is_empty() {
            continue;
        }
        let starts_word = s.starts_with(BPE);
        let s = s.trim_matches(BPE);
        if starts_word && !out.is_empty() {
            out.push(' ');
        }
        out.push_str(s);
        prev = Some(best);
    }
    out.chars()
        .filter(|c| !('\u{E000}'..='\u{E0FF}').contains(c))
        .collect()
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
        .map(|t| {
            (0..log_probs.shape()[1])
                .map(|v| log_probs[[t, v]])
                .collect()
        })
        .collect();
    let text = dec.decode(&rows, &hw, beam_width, -5.0, -10.0);
    text.chars()
        .filter(|c| !('\u{E000}'..='\u{E0FF}').contains(c))
        .collect::<String>()
}

/// Load a hotword list (one word/phrase per line; blank lines ignored).
pub fn load_hotwords(path: &str) -> Result<Vec<String>> {
    let txt = std::fs::read_to_string(path)?;
    Ok(parse_hotwords(&txt))
}

/// Parse the same newline-oriented hotword format accepted by the CLI file and
/// the optional multipart `hotwords` field.
pub fn parse_hotwords(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn greedy_joins_bpe_pieces_and_resets_after_blank() {
        let labels = vec!["▁شن".into(), "ا".into(), "".into()];
        let probs = array![
            [0.0, -4.0, -4.0],
            [-4.0, 0.0, -4.0],
            [-4.0, -4.0, 0.0],
            [-4.0, 0.0, -4.0],
        ];
        assert_eq!(greedy(&probs, &labels, 2), "شناا");
    }

    #[test]
    fn greedy_and_hotbeam_do_not_space_every_persian_bpe_piece() {
        // Regression for the old renderer, which turned these tokens into
        // "فرو ش نده سی ب" by inserting a space after every CTC token.
        let labels = vec![
            "▁فرو".into(),
            "ش".into(),
            "نده".into(),
            "▁سی".into(),
            "ب".into(),
            "▁رو".into(),
            "ست".into(),
            "ای".into(),
            "▁کوچ".into(),
            "ک".into(),
            "".into(),
        ];
        let path = [
            0, 10, 1, 10, 2, 10, 3, 10, 4, 10, 5, 10, 6, 10, 7, 10, 8, 10, 9,
        ];
        let mut probs = Array2::from_elem((path.len(), labels.len()), -20.0);
        for (frame, token) in path.into_iter().enumerate() {
            probs[[frame, token]] = 0.0;
        }

        let expected = "فروشنده سیب روستای کوچک";
        assert_eq!(greedy(&probs, &labels, 10), expected);
        assert_eq!(decode_hotword(&probs, &labels, &[], 2.5, 20), expected);
    }

    #[test]
    fn hotword_parser_ignores_comments_and_blank_lines() {
        assert_eq!(
            parse_hotwords("# names\nشنوا\n\n  رضا  \n"),
            vec!["شنوا", "رضا"]
        );
    }
}
