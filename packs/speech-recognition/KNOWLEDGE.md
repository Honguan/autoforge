# speech-recognition — knowledge for AI planners

## When to use
- Speech-to-text where base Whisper makes systematic mistakes: accents, domain vocabulary (medical, product names),
  call-centre audio, a language Whisper handles poorly (Taiwanese-accented Mandarin, Cantonese, regional languages).
- You have recordings **segmented into clips of ≤ 30 s** with exact transcripts. 1 h of audio gives visible gains,
  10 h+ gives strong ones. Always look at `baseline_wer` first: if it is already low, fine-tuning adds little.

## When NOT to use
- Long recordings without aligned segment transcripts → segment them first (VAD + alignment), then train.
- Speaker labels (diarization), word timestamps, or real-time streaming → not covered by this pack.
- Classifying sounds (alarm / no alarm) → an audio-classification model, not ASR.

## Data format
CSV with columns `audio,text` (aliases: `path`/`file` and `transcript`/`sentence`). Audio paths are relative to
the CSV's folder. wav / flac / ogg; any sample rate and channel count (mixed to mono, resampled to 16 kHz).
```
audio,text
clips/0001.wav,請問明天的會議幾點開始
clips/0002.flac,The invoice number is A-1042.
```
Clips > 30 s are truncated (warning in the log) — their text will no longer match; split them instead.

## Recipe
`whisper_finetune`: full fine-tuning of `WhisperForConditionalGeneration` with AdamW (constant LR), loss on the
transcript tokens. Model sizes: `openai/whisper-tiny` (39M, CPU ok), `-base` (74M), `-small` (244M, default,
~8 GB VRAM at batch 8), `-medium` (769M, 24 GB+ or `freeze_encoder: true`). `*.en` models are English-only.
`language`: ISO code of the data ("zh", "en", "ja", "yue"...) — set it; `null` detects it once from a few
training clips. `task`: `transcribe` (same language) or `translate` (to English).

## Metrics
- `wer` (word error rate, HPO metric, **minimize**), `cer` (character error rate), `val_loss`, per epoch;
  lowercased, punctuation removed. `baseline_wer` / `baseline_cer` = zero-shot base model, logged before training.
- **Chinese / Japanese / Thai** (no spaces): WER treats a whole sentence as one word — optimise `cer` instead
  (`hpo start ... -m cer`) and report cer.
- WER can exceed 1.0 when the model hallucinates extra words.

## Tuning guide
- `learning_rate` 1e-5 (small) / 6e-6 (medium) / 3e-5 (tiny/base); higher often makes WER worse than baseline.
- `epochs` 2–5; stop when `wer` rises again (best epoch is kept automatically).
- Effective batch = `batch_size × grad_accum`; aim for 16–32. On OOM halve `batch_size` and double `grad_accum`.
- `freeze_encoder: true` halves memory and is safer with < 1 h of data or when only vocabulary/spelling changes.
- `num_beams: 1` for fast evaluation; 4–5 gives slightly lower WER at serving time.
- `max_label_length` 225 tokens (~30 s of speech); lower only if transcripts are short.

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| `audio file not found: ...` | paths not relative to the CSV's folder | fix the paths in the CSV |
| `need an audio column ... found columns` | column names not recognised | rename to `audio`,`text` |
| wer > baseline_wer after epoch 0 | learning rate too high / too little data | lower `learning_rate`, `freeze_encoder: true` |
| repeated words / hallucinated endings | clips truncated or transcripts do not match audio | split > 30 s clips; check transcripts |
| wrong language or translated output | `language` not set, mixed-language data | set `language` explicitly |
| CUDA out of memory | model / batch too large | `batch_size: 4`, `grad_accum: 4`, smaller model, `freeze_encoder` |
| very slow on CPU | Whisper is heavy | use a GPU, or `whisper-tiny`/`whisper-base` |

## Serving
- Predict row `{"audio": "<path on the server>"}` or base64 / data URL of a wav/flac/ogg file →
  `{"prediction": "transcript"}` (plus `"warning"` if the clip was longer than 30 s and got truncated).
- The output dir is a standard Hugging Face Whisper dir (model + processor + generation config with the language
  and task set), usable with `transformers.pipeline("automatic-speech-recognition", model=dir)`.
