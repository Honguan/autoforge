"""Smoke test on CPU: a tiny random Whisper built from a config (no weights download) with the real
openai/whisper-tiny processor and generation config (a few MB, downloaded once), on synthetic sine-wave clips."""

import base64
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("soundfile")
pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("whisper_train", ROOT / "trainers" / "whisper_train.py")
wt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wt)
WORDS = ["one", "two", "three", "four", "five", "six"]


@pytest.fixture(scope="module")
def tiny_whisper(tmp_path_factory):
    from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    gen = GenerationConfig.from_pretrained("openai/whisper-tiny")
    gen.suppress_tokens, gen.begin_suppress_tokens = [], []  # keeps random-model decoding fast on CPU
    cfg = WhisperConfig(vocab_size=len(proc.tokenizer), d_model=16, encoder_layers=1, decoder_layers=1,
                        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                        num_mel_bins=80, max_source_positions=1500, max_target_positions=64,
                        decoder_start_token_id=gen.decoder_start_token_id, pad_token_id=gen.pad_token_id,
                        bos_token_id=gen.bos_token_id, eos_token_id=gen.eos_token_id)
    model = WhisperForConditionalGeneration(cfg)
    model.generation_config = gen
    d = tmp_path_factory.mktemp("tiny-whisper")
    model.save_pretrained(d)
    proc.save_pretrained(d)
    return str(d)


def _data(tmp_path, sr=22050):
    import numpy as np
    import soundfile as sf

    clips = tmp_path / "clips"
    clips.mkdir()
    lines = ["file,transcript"]
    for i, w in enumerate(WORDS):
        t = np.arange(sr) / sr
        stereo = np.stack([np.sin(2 * np.pi * (220 + 110 * i) * t)] * 2, axis=1) * 0.3
        sf.write(clips / f"{w}.{'flac' if i % 2 else 'wav'}", stereo.astype("float32"), sr)
        lines.append(f"clips/{w}.{'flac' if i % 2 else 'wav'},{w.capitalize()}.")
    (tmp_path / "data.csv").write_text("\n".join(lines))
    return tmp_path / "data.csv"


def test_train_predict(tmp_path, tiny_whisper):
    from autoforge.engine.context import TrainContext

    data = _data(tmp_path)
    hp = {"model_name": tiny_whisper, "language": None, "task": "transcribe", "epochs": 2, "batch_size": 3,
          "grad_accum": 1, "learning_rate": 0.001, "max_label_length": 6, "val_fraction": 0.2, "num_beams": 1,
          "freeze_encoder": True}
    ctx = TrainContext(tmp_path / "run", str(data), None, "speech-to-text", hp, "cpu")
    res = wt.train(ctx)
    assert res["metrics"]["wer"] >= 0 and 0 <= res["metrics"]["cer"] and res["metrics"]["val_loss"] > 0
    names = {json.loads(x).get("name") for x in (tmp_path / "run" / "events.jsonl").read_text().splitlines()}
    assert {"baseline_wer", "train_loss", "wer", "cer", "val_loss"} <= names
    out = Path(res["model_path"])
    assert (out / "config.json").is_file() and any((out / f).is_file() for f in ("processor_config.json", "preprocessor_config.json"))
    assert (out / "io.json").is_file() and json.loads((out / "asr.json").read_text())["task"] == "transcribe"
    state = json.loads((tmp_path / "run" / "checkpoints" / "state.json").read_text())
    assert state["epoch"] == 1 and state["language"]

    m = wt.load_model(out)
    b64 = base64.b64encode((tmp_path / "clips" / "two.flac").read_bytes()).decode()
    pred = wt.predict(m, [{"audio": str(tmp_path / "clips" / "one.wav")}, {"audio": "data:audio/flac;base64," + b64}])
    assert all(isinstance(p["prediction"], str) for p in pred)


def test_error_rates():
    r = wt.error_rates(["Hello, world!", "你好世界"], ["hello word", "你好世"])
    assert r["wer"] == pytest.approx(2 / 3)  # "world"->"word" sub (1) + "你好世界"->"你好世" sub (1) of 3 words
    assert r["cer"] == pytest.approx(2 / 14)  # one deleted letter + one deleted 字 of 10 + 4 characters


def test_missing_audio_file_is_named(tmp_path):
    (tmp_path / "data.csv").write_text("audio,text\nclips/a.wav,hello\nclips/b.wav,world\n")
    with pytest.raises(FileNotFoundError, match="a.wav"):
        wt.read_manifest(str(tmp_path / "data.csv"))
    (tmp_path / "bad.csv").write_text("foo,bar\nx,y\n")
    with pytest.raises(ValueError, match="found columns"):
        wt.read_manifest(str(tmp_path / "bad.csv"))
