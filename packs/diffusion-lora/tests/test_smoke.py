"""Smoke test on CPU with a tiny random Stable Diffusion pipeline (downloads ~3 MB once)."""

import base64
import importlib.util
import io
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("peft")
PIL = pytest.importorskip("PIL.Image")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sd_lora", ROOT / "trainers" / "sd_lora.py")
sd_lora = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd_lora)
TINY = "hf-internal-testing/tiny-sd-pipe"
torch.set_num_threads(1)  # tensors are tiny: thread start-up costs far more than the maths on small CPU quotas


def _data(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for i, colour in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        PIL.new("RGB", (48, 40), colour).save(d / f"img{i}.png")
    (d / "img0.txt").write_text("a red square")
    return d


def test_sd_lora_tiny(tmp_path):
    from autoforge.engine.context import TrainContext

    hp = {"model_name": TINY, "resolution": 32, "epochs": 2, "batch_size": 2, "grad_accum": 1, "lora_r": 4,
          "lora_alpha": 4, "instance_prompt": "a photo of sks square", "validation_prompt": "a sks square",
          "num_validation_images": 1, "num_inference_steps": 2}
    ctx = TrainContext(tmp_path / "run", str(_data(tmp_path)), None, "text-to-image", hp, "cpu")
    res = sd_lora.train(ctx)
    out = Path(res["model_path"])
    assert (out / "pytorch_lora_weights.safetensors").is_file() and (out / "io.json").is_file()
    assert (out / "samples" / "sample_0.png").is_file()
    assert res["metrics"]["val_loss"] > 0 and res["metrics"]["steps"] == 4
    m = sd_lora.load_model(out)
    pred = sd_lora.predict(m, [{"prompt": "a sks square", "num_inference_steps": 2, "seed": 7}])
    assert pred[0]["seed"] == 7 and pred[0]["prediction"].startswith("data:image/png;base64,")
    img = PIL.open(io.BytesIO(base64.b64decode(pred[0]["prediction"].split(",", 1)[1])))
    assert img.size == (32, 32)
    # resume after the last epoch keeps the best adapter and does not retrain
    again = sd_lora.train(TrainContext(tmp_path / "run", ctx.data_path, None, "text-to-image",
                                       dict(hp, validation_prompt=None), "cpu"))
    assert again["metrics"]["val_loss"] == res["metrics"]["val_loss"]


def test_bad_data_is_explained(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    PIL.new("RGB", (8, 8)).save(d / "a.png")
    with pytest.raises(ValueError, match="no caption"):
        sd_lora._scan(d, None)
    (d / "metadata.csv").write_text("file,caption\na.png,x\n")
    with pytest.raises(ValueError, match="file_name,text"):
        sd_lora._scan(d, None)
    with pytest.raises(FileNotFoundError):
        sd_lora._scan(tmp_path / "missing", "x")
