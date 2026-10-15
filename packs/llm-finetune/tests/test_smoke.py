"""Smoke test on CPU with a tiny random Llama (downloads a few MB once)."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sft", ROOT / "trainers" / "sft.py")
sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft)
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _data(tmp_path):
    p = tmp_path / "chat.jsonl"
    rows = [{"prompt": f"Say the number {i}", "response": f"The number is {i}."} for i in range(12)]
    rows.append({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]})
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_sft_lora_tiny(tmp_path):
    from autoforge.engine.context import TrainContext

    ctx = TrainContext(tmp_path / "run", str(_data(tmp_path)), None, "llm-sft",
                       {"model_name": TINY, "epochs": 2, "batch_size": 4, "grad_accum": 1, "max_length": 64,
                        "learning_rate": 0.001, "lora_r": 4, "lora_alpha": 8}, "cpu")
    res = sft.train(ctx)
    out = Path(res["model_path"])
    assert (out / "adapter_config.json").is_file() and (out / "io.json").is_file()
    assert res["metrics"]["val_loss"] > 0
    m = sft.load_model(out)
    pred = sft.predict(m, [{"prompt": "Say the number 3", "max_new_tokens": 5}])
    assert isinstance(pred[0]["prediction"], str)
    merged = sft.export_merged(out, tmp_path / "merged", {})
    assert any(Path(merged).glob("*.safetensors"))


def test_bad_data_is_explained(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"text": "no prompt"}) + "\n" + json.dumps({"text": "x"}))
    with pytest.raises(ValueError, match="needs 'prompt'"):
        sft._read_examples(str(p))
