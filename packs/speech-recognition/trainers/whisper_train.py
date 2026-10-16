"""Fine-tune Whisper (speech-to-text) on your own recordings in a plain PyTorch loop.

Data: a CSV with one clip per row:
  audio,text                 audio = path to a wav/flac/ogg file, relative to the CSV's folder (or absolute)
  Also accepted: path/file/audio_path for the audio column, transcript/sentence/transcription for the text.
Audio is mixed to mono and resampled to 16 kHz; clips longer than 30 s are truncated (with a log warning) because
Whisper sees at most 30 s per clip.
Language: `language` (e.g. "zh", "en") fixes the decoder prompt; null = detected once by the base model on a few
training clips, then kept fixed for training, evaluation and serving.
Metrics: wer / cer (own edit distance on lowercased text without punctuation), val_loss; baseline_wer / baseline_cer
are the zero-shot scores of the base model, logged before training.
AutoForge controls: check_stop per micro-batch, live learning_rate via check_updates, checkpoint per epoch
(save_pretrained + checkpoints/state.json, resumed by `run retry`), the epoch with the lowest wer is kept.
Output: a standard Hugging Face dir (WhisperForConditionalGeneration + WhisperProcessor) usable by pipelines.
"""

from __future__ import annotations

import base64
import io
import json
import math
import random
import re
import shutil
import unicodedata
from pathlib import Path

SR = 16000
MAX_SECONDS = 30
AUDIO_COLS = ["audio", "path", "file", "audio_path", "filename"]
TEXT_COLS = ["text", "transcript", "sentence", "transcription"]


# ---------- data ----------

def _pick(cols, names):
    lower = {str(c).strip().lower(): c for c in cols}
    return next((lower[n] for n in names if n in lower), None)


def read_manifest(path: str) -> list[tuple[Path, str]]:
    """Read (audio file, transcript) pairs from the CSV; audio paths are resolved against the CSV's folder."""
    import pandas as pd

    p = Path(path)
    if p.is_dir():
        found = sorted(p.glob("*.csv"))
        if len(found) != 1:
            raise FileNotFoundError(f"{p} is a folder; point data_path at the CSV file (audio,text)")
        p = found[0]
    if not p.is_file():
        raise FileNotFoundError(f"data file not found: {p} (expected a CSV with audio,text columns)")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    a, t = _pick(df.columns, AUDIO_COLS), _pick(df.columns, TEXT_COLS)
    if a is None or t is None:
        raise ValueError(f"{p.name}: need an audio column ({'/'.join(AUDIO_COLS)}) and a text column "
                         f"({'/'.join(TEXT_COLS)}); found columns {list(df.columns)}")
    items = []
    for n, (f, text) in enumerate(zip(df[a], df[t]), 2):
        f, text = f.strip(), text.strip()
        if not f or not text:
            continue
        fp = Path(f) if Path(f).is_absolute() else p.parent / f
        if not fp.is_file():
            raise FileNotFoundError(f"{p.name} line {n}: audio file not found: {fp}")
        items.append((fp, text))
    if len(items) < 2:
        raise ValueError(f"{p.name}: need at least 2 rows with both audio and text, found {len(items)}")
    return items


def load_audio(src) -> tuple:
    """Return (16 kHz mono float32 array, original seconds). `src` is a path, base64 or data URL of a wav/flac/ogg."""
    import numpy as np
    import soundfile as sf

    if isinstance(src, (bytes, bytearray)):
        data, sr = sf.read(io.BytesIO(src), dtype="float32", always_2d=True)
    else:
        s = str(src)
        if s.startswith("data:") or (len(s) > 260 and not Path(s).exists()):
            try:
                raw = base64.b64decode(s.split(",", 1)[-1], validate=False)
            except ValueError as e:
                raise ValueError(f"audio is neither a file path nor valid base64: {e}") from None
            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        else:
            if not Path(s).is_file():
                raise FileNotFoundError(f"audio file not found: {s}")
            data, sr = sf.read(s, dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    seconds = len(x) / SR
    return x[:SR * MAX_SECONDS], seconds


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def _edits(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def error_rates(refs: list[str], hyps: list[str]) -> dict:
    """Corpus WER (words) and CER (characters, spaces ignored) after lowercasing and removing punctuation."""
    we = wn = ce = cn = 0
    for r, h in zip(refs, hyps):
        r, h = _normalize(r), _normalize(h)
        we += _edits(r.split(), h.split())
        wn += len(r.split())
        ce += _edits(list(r.replace(" ", "")), list(h.replace(" ", "")))
        cn += len(r.replace(" ", ""))
    return {"wer": we / max(wn, 1), "cer": ce / max(cn, 1)}


# ---------- model helpers ----------

def _features(proc, clips, device):
    feats = proc.feature_extractor([c for c in clips], sampling_rate=SR, return_tensors="pt").input_features
    return feats.to(device)


def _labels(proc, texts, max_len, start_id):
    import torch

    rows = []
    for t in texts:
        ids = proc.tokenizer(t).input_ids  # <|startoftranscript|><|lang|><|task|><|notimestamps|> text <|endoftext|>
        if ids and ids[0] == start_id:  # the model prepends decoder_start_token_id itself
            ids = ids[1:]
        if len(ids) > max_len:
            ids = ids[:max_len - 1] + [ids[-1]]
        rows.append(ids)
    out = torch.full((len(rows), max(len(r) for r in rows)), -100)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r)
    return out


def _gen_kwargs(model, lang, task, num_beams, max_new):
    kw = {"num_beams": int(num_beams), "max_new_tokens": int(max_new)}
    if getattr(model.generation_config, "is_multilingual", False) and getattr(model.generation_config, "lang_to_id", None):
        if lang:
            kw["language"] = lang
        kw["task"] = task
    return kw


def _transcribe(model, proc, clips, gen_kw, device, bs):
    import torch

    model.eval()
    texts = []
    with torch.no_grad():
        for i in range(0, len(clips), bs):
            ids = model.generate(_features(proc, clips[i:i + bs], device), **gen_kw)
            texts += [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
    return texts


def _evaluate(model, proc, val_clips, val_texts, gen_kw, device, bs, max_len, start_id):
    import torch

    hyps = _transcribe(model, proc, val_clips, gen_kw, device, bs)
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val_clips), bs):
            labels = _labels(proc, val_texts[i:i + bs], max_len, start_id).to(device)
            loss = model(input_features=_features(proc, val_clips[i:i + bs], device), labels=labels).loss
            total, n = total + float(loss) * labels.shape[0], n + labels.shape[0]
    return {**error_rates(val_texts, hyps), "val_loss": total / max(n, 1)}, hyps


def _detect_language(model, proc, clips, device):
    import torch

    gc = model.generation_config
    lang_to_id = getattr(gc, "lang_to_id", None)
    if not getattr(gc, "is_multilingual", False) or not lang_to_id:
        return None
    with torch.no_grad():
        ids = model.detect_language(_features(proc, clips, device)).view(-1).tolist()
    id_to_lang = {v: k.strip("<|>") for k, v in lang_to_id.items()}
    langs = [id_to_lang.get(i) for i in ids if i in id_to_lang]
    return max(set(langs), key=langs.count) if langs else None


# ---------- training ----------

def train(ctx) -> dict:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    hp = ctx.hyperparameters
    random.seed(ctx.seed)
    torch.manual_seed(ctx.seed)
    cuda = ctx.device == "cuda" and torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    base = hp.get("model_name", "openai/whisper-small")
    task = hp.get("task") or "transcribe"
    if task not in ("transcribe", "translate"):
        raise ValueError(f"task must be 'transcribe' or 'translate', got {task!r}")
    bs, accum, epochs = int(hp.get("batch_size", 8)), int(hp.get("grad_accum", 2)), int(hp.get("epochs", 3))
    beams, max_len = int(hp.get("num_beams", 1)), int(hp.get("max_label_length", 225))

    items = read_manifest(ctx.data_path)
    random.shuffle(items)
    n_val = min(len(items) - 1, max(1, int(len(items) * float(hp.get("val_fraction", 0.1)))))
    val_items, train_items = items[:n_val], items[n_val:]
    import soundfile as sf

    long = [f.name for f, _ in items if sf.info(str(f)).duration > MAX_SECONDS]
    if long:
        ctx.log(f"warning: {len(long)} clip(s) longer than {MAX_SECONDS} s are truncated (Whisper sees 30 s max), "
                f"e.g. {long[:3]}; split long recordings into segments with matching text")
    val_clips = [load_audio(f)[0] for f, _ in val_items]
    val_texts = [t for _, t in val_items]

    state_file = ctx.latest_checkpoint()
    state = json.loads(state_file.read_text()) if state_file is not None and state_file.name == "state.json" else None
    src = state["path"] if state else base
    proc = WhisperProcessor.from_pretrained(src)
    model = WhisperForConditionalGeneration.from_pretrained(src).to(device)
    max_len = min(max_len, int(model.config.max_target_positions) - 4)
    start_id = int(model.config.decoder_start_token_id)

    if state:
        start_epoch, best, lang = state["epoch"] + 1, state.get("best"), state.get("language")
        ctx.log(f"resuming after epoch {state['epoch']}")
    else:
        start_epoch, best, lang = 0, None, hp.get("language") or None
        if lang is None:
            lang = _detect_language(model, proc, [load_audio(f)[0] for f, _ in train_items[:8]], device)
            if lang:
                ctx.log(f"language not set: detected '{lang}' on the training clips (set `language` to override)")
    multilingual = bool(getattr(model.generation_config, "is_multilingual", False))
    if multilingual:
        proc.tokenizer.set_prefix_tokens(language=lang, task=task, predict_timestamps=False)
        model.generation_config.language, model.generation_config.task = lang, task
        model.generation_config.forced_decoder_ids = None
    gen_kw = _gen_kwargs(model, lang, task, beams, max_len)

    if hp.get("freeze_encoder"):
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ctx.log(f"{base} on {device}: {len(train_items)} train / {len(val_items)} val clips, language={lang or 'n/a'}, "
            f"task={task}, {trainable:,} trainable parameters{' (encoder frozen)' if hp.get('freeze_encoder') else ''}")
    if not cuda:
        ctx.log("note: running on CPU; Whisper fine-tuning is slow without a GPU (use whisper-tiny/base for tests)")

    if start_epoch == 0:
        m0, _ = _evaluate(model, proc, val_clips, val_texts, gen_kw, device, bs, max_len, start_id)
        ctx.log_metric("baseline_wer", m0["wer"], step=0)
        ctx.log_metric("baseline_cer", m0["cer"], step=0)
        ctx.log(f"zero-shot baseline: wer={m0['wer']:.4f} cer={m0['cer']:.4f}")

    amp = cuda and torch.cuda.is_bf16_supported()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(hp.get("learning_rate", 1e-5)), weight_decay=0.01)
    step = start_epoch * math.ceil(math.ceil(len(train_items) / bs) / accum)  # optimizer steps already done
    for epoch in range(start_epoch, epochs):
        model.train()
        random.Random(ctx.seed + epoch).shuffle(train_items)
        opt.zero_grad()
        for i in range(0, len(train_items), bs):
            ctx.check_stop()
            changed = ctx.check_updates()
            if "learning_rate" in changed:
                for g in opt.param_groups:
                    g["lr"] = float(changed["learning_rate"])
            chunk = train_items[i:i + bs]
            feats = _features(proc, [load_audio(f)[0] for f, _ in chunk], device)
            labels = _labels(proc, [t for _, t in chunk], max_len, start_id).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = model(input_features=feats, labels=labels).loss
            (loss / accum).backward()
            if (i // bs + 1) % accum == 0 or i + bs >= len(train_items):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                ctx.log_metric("train_loss", loss.item(), step=step)
                step += 1
        m, hyps = _evaluate(model, proc, val_clips, val_texts, gen_kw, device, bs, max_len, start_id)
        for k, v in m.items():
            ctx.log_metric(k, v, step=epoch)
        ctx.log(f"epoch {epoch}: wer={m['wer']:.4f} cer={m['cer']:.4f} val_loss={m['val_loss']:.4f} | "
                f"e.g. {val_texts[0]!r} -> {hyps[0]!r}")
        ck = ctx.checkpoint_dir / f"epoch{epoch}"
        model.save_pretrained(ck)
        proc.save_pretrained(ck)
        if best is None or m["wer"] <= best["metrics"]["wer"]:
            old, best = best, {"epoch": epoch, "path": str(ck), "metrics": m}
            if old and Path(old["path"]).is_dir():
                shutil.rmtree(old["path"], ignore_errors=True)
        (ctx.checkpoint_dir / "state.json").write_text(json.dumps(
            {"epoch": epoch, "path": str(ck), "best": best, "language": lang}))
        prev = ctx.checkpoint_dir / f"epoch{epoch - 1}"  # keep only the latest and the best epoch on disk
        if epoch > 0 and prev.is_dir() and str(prev) != best["path"]:
            shutil.rmtree(prev, ignore_errors=True)

    out = Path(ctx.output_dir)
    if best and Path(best["path"]).is_dir():
        shutil.copytree(best["path"], out, dirs_exist_ok=True)
        metrics = best["metrics"]
        ctx.log(f"kept epoch {best['epoch']} (wer={metrics['wer']:.4f})")
    else:  # nothing trained (resumed after the last epoch)
        metrics, _ = _evaluate(model, proc, val_clips, val_texts, gen_kw, device, bs, max_len, start_id)
        model.save_pretrained(out)
        proc.save_pretrained(out)
    (out / "asr.json").write_text(json.dumps({"language": lang, "task": task, "num_beams": beams,
                                              "max_new_tokens": max_len}))
    (out / "io.json").write_text(json.dumps({"features": [
        {"name": "audio", "dtype": "text",
         "description": "wav/flac/ogg file path on the server, or base64 / data URL of the file; max 30 s"}]}))
    return {"metrics": dict(metrics), "model_path": str(out)}


# ---------- serving ----------

def load_model(model_dir):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    d = Path(model_dir)
    cfg = json.loads((d / "asr.json").read_text()) if (d / "asr.json").is_file() else {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(d).to(device).eval()
    proc = WhisperProcessor.from_pretrained(d)
    gen_kw = _gen_kwargs(model, cfg.get("language"), cfg.get("task", "transcribe"), cfg.get("num_beams", 1),
                         cfg.get("max_new_tokens", 225))
    return {"model": model, "proc": proc, "gen_kw": gen_kw, "device": device}


def predict(m, rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if not r.get("audio"):
            raise ValueError("each row needs 'audio': a wav/flac/ogg file path or base64 of the file")
        clip, seconds = load_audio(r["audio"])
        text = _transcribe(m["model"], m["proc"], [clip], m["gen_kw"], m["device"], 1)[0]
        res = {"prediction": text}
        if seconds > MAX_SECONDS:
            res["warning"] = f"audio is {seconds:.0f} s; only the first {MAX_SECONDS} s were transcribed"
        out.append(res)
    return out
