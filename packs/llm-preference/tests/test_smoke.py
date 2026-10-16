"""Smoke test on CPU with a tiny random Llama (downloads a few MB once)."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dpo", ROOT / "trainers" / "dpo.py")
dpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dpo)
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _data(tmp_path):
    p = tmp_path / "prefs.jsonl"
    rows = [{"prompt": f"What is {i} plus 1?", "chosen": f"{i + 1}.", "rejected": f"I don't know, maybe {i * 7}?"}
            for i in range(11)]
    rows.append({"prompt": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
                 "chosen": "Hello!", "rejected": "Greetings, it is a pleasure to make your acquaintance today."})
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_dpo_lora_tiny(tmp_path):
    from autoforge.engine.context import TrainContext

    hp = {"model_name": TINY, "epochs": 2, "batch_size": 4, "grad_accum": 1, "max_length": 64,
          "max_prompt_length": 32, "learning_rate": 0.001, "lora_r": 4, "lora_alpha": 8, "beta": 0.1,
          "eval_fraction": 0.25}
    ctx = TrainContext(tmp_path / "run", str(_data(tmp_path)), None, "llm-preference", hp, "cpu")
    res = dpo.train(ctx)
    out = Path(res["model_path"])
    assert (out / "adapter_config.json").is_file() and (out / "io.json").is_file()
    m = res["metrics"]
    assert 0.0 <= m["reward_accuracy"] <= 1.0 and m["val_loss"] > 0 and "reward_margin" in m
    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert {"train_loss", "reward_accuracy"} <= {e.get("name") for e in events}
    model = dpo.load_model(out)
    pred = dpo.predict(model, [{"prompt": "What is 3 plus 1?", "max_new_tokens": 5}])
    assert isinstance(pred[0]["prediction"], str)
    merged = dpo.export_merged(out, tmp_path / "merged", {})
    assert any(Path(merged).glob("*.safetensors"))


def test_bad_data_is_explained(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"prompt": "x", "response": "y"}) + "\n" + json.dumps({"prompt": "a", "chosen": "b"}))
    with pytest.raises(ValueError, match="needs 'prompt', 'chosen' and 'rejected'"):
        dpo._read_pairs(str(p))
    with pytest.raises(FileNotFoundError):
        dpo._read_pairs(str(tmp_path / "missing.jsonl"))
