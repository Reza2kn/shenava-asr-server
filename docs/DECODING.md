# Improving Shenava word accuracy

**English** | [فارسی](DECODING.fa.md)

The acoustic backend (tract or CoreML) produces the same CTC scores. Accuracy changes belong in
decoding, normalization, or model choice—not in backend-specific code.

## 1. Start with the two returned transcripts

- `greedy` is the acoustic argmax baseline.
- `text` is hotbeam output when any startup or per-request hotwords are present; otherwise it is
  identical to `greedy`.

Log both during evaluation. A hotword configuration is useful only if it improves the target-word
errors without causing unacceptable substitutions elsewhere.

## 2. Use Shenava hotbeam for names and domain terms

Create a UTF-8 file with one word or phrase per line. Blank lines and lines beginning with `#` are
ignored.

```text
# people
رضا سیار

# product/domain vocabulary
شنوا
هم‌نویسه
```

```bash
./run.sh --hotwords hotwords.txt --beam 80 --hotword-weight 2.5
```

Hotwords can also be supplied per request through the multipart `hotwords` field. Phrases are
split into word-level boosts by `shenava-ctc-beam`; they are not treated as a forced phrase. This is
intentional: the decoder keeps the acoustic score and adds full-word and partial-prefix credit.

Tune on a frozen representative set:

1. Record greedy WER plus a target-term recall/error measure.
2. Sweep a small set of weights around `2.5`; do not assume the beam crate's standalone default of
   `10.0` is right for every list.
3. Test each beam width/weight pair on the same utterances.
4. Admit a setting only if the target improvement survives your general non-regression gate.

Large, generic dictionaries make nearly every path look like a hotword and usually reduce the
benefit. Use short request-specific lists when the application already knows likely names, places,
contacts, medications, or catalog terms.

## 3. Bringing your own language model

There is deliberately no `--lm model.arpa` flag today. `shenava-ctc-beam` implements the
`pyctcdecode` **no-LM path** and returns one best transcript. Describing it as KenLM-compatible
would be inaccurate.

There are two sound integration patterns:

### LM inside CTC beam search

Extend `shenava-ctc-beam` with a scorer called whenever a BPE word boundary folds `word_part` into
the completed text. Rank beams with a combined score such as:

```text
acoustic_logp + hotword_score + alpha * lm_logp + beta * completed_word_count
```

The scorer should be a Rust trait so an embedded n-gram, neural LM, or an FFI-backed user runtime
can implement the same boundary. Cache LM states per beam prefix; calling an HTTP model inside each
frame expansion will be far too slow. Keep the acoustic log probability separate so merged CTC
paths are still combined correctly.

### N-best second-pass rescoring

First extend the beam crate to return N completed hypotheses and acoustic scores. Batch those N
hypotheses through the user's LM, combine the scores, then return the winner. This is the cleanest
boundary for a Go-hosted or remote LM because it needs one batched request per utterance rather
than requests inside the frame loop.

For either approach, the LM must match Shenava's Persian text conventions:

- normalize Arabic/Persian `ی` and `ک` consistently;
- decide how zero-width non-joiners and punctuation are represented;
- score words after ve_tok_v4 BPE pieces have been joined;
- handle BOS/EOS and unknown words explicitly;
- know whether LM scores are natural log or base-10 before choosing `alpha`;
- tune `alpha` and `beta` on held-out audio, not on the LM training text.

The server-side integration point is `src/decode.rs`: it receives backend-independent
`Array2<f32>` CTC scores. A custom decoder can be added there without touching tract, CoreML,
fbank, the HTTP upload path, or the Go client.

## 4. Other improvements that do not pretend to be an LM

- Context-derived hotwords: contacts, current document terms, meeting attendees, product catalog.
- A stronger Shenava acoustic model, evaluated on the same fixed benchmark.
- Persian inverse text normalization for display. This changes formatting and may improve a
  display-oriented metric, but it does not recover an acoustically wrong word.
- Punctuation/casing restoration as a separate post-process. Keep raw ASR text available so it is
  not confused with recognition accuracy.
