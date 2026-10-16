"""Verify the GPU packs on a real NVIDIA GPU: `autoforge eval gpu`.

Each check installs a pack, writes a small dataset that a correct trainer learns easily, trains a
real pretrained model through the normal engine (worker process, plan validation, checkpoints) with
device=cuda, samples the GPU memory while it runs, then registers the model and predicts once.

Tiers: `quick` uses small real models (minutes per check, ~5 GB of downloads in total);
`full` uses the recipes' production-size defaults (QLoRA on a 1.5B model, whisper-small, SD 1.5 at 512 px).
Everything runs in a throw-away AUTOFORGE_HOME; the report is kept in ~/.autoforge/evals/.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from autoforge import config

# ---------------------------------------------------------------- data (needs the [deep] extra: numpy, pillow)


def _img(path: Path, size: int, color: tuple[int, int, int], rng: random.Random, shapes=()) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    arr = np.clip(np.array(color) + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 12, (size, size, 3)), 0, 255)
    im = Image.fromarray(arr.astype("uint8"))
    draw = ImageDraw.Draw(im)
    for kind, box, fill in shapes:
        (draw.rectangle if kind == "rect" else draw.ellipse)(box, fill=fill)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def data_images(d: Path) -> Path:
    rng = random.Random(1)
    for cls, color in [("red", (210, 40, 40)), ("green", (40, 180, 60)), ("blue", (40, 70, 210))]:
        for i in range(40):
            _img(d / "imgs" / cls / f"{i}.png", 96, color, rng)
    return d / "imgs"


def data_reviews(d: Path) -> Path:
    rng = random.Random(2)
    good = ["great quality, works perfectly", "fast delivery and friendly support", "love it, highly recommended",
            "很好用，非常滿意", "品質很好，會再回購"]
    bad = ["broke after one day", "terrible support, never again", "waste of money, do not buy",
           "收到就壞了", "品質很差，要退貨"]
    rows = ["text,label"] + [f"\"{rng.choice(good if i % 2 else bad)} ({i})\",{'pos' if i % 2 else 'neg'}" for i in range(240)]
    p = d / "reviews.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def data_chat(d: Path) -> Path:
    rows = [{"prompt": f"What is {a} plus {b}?", "response": f"{a} plus {b} is {a + b}."}
            for a in range(1, 9) for b in range(1, 7)]
    p = d / "chat.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def data_prefs(d: Path) -> Path:
    rows = [{"prompt": f"Reply politely to a customer asking about order {i}.",
             "chosen": f"Thank you for your patience! Order {i} will arrive within three days.",
             "rejected": f"Order {i}? Just wait."} for i in range(40)]
    p = d / "prefs.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def data_boxes(d: Path) -> Path:
    rng = random.Random(3)
    root = d / "boxes"
    lines = ["image,xmin,ymin,xmax,ymax,label"]
    for i in range(48):
        x, y = rng.randint(8, 60), rng.randint(8, 60)
        cls, fill = ("square", (230, 200, 30)) if i % 2 else ("disc", (30, 200, 230))
        _img(root / "images" / f"{i}.png", 160, (90, 90, 90), rng, [("rect" if cls == "square" else "ellipse",
                                                                       (x, y, x + 70, y + 70), fill)])
        lines.append(f"images/{i}.png,{x},{y},{x + 70},{y + 70},{cls}")
    (root / "annotations.csv").write_text("\n".join(lines) + "\n")
    return root


def data_masks(d: Path) -> Path:
    from PIL import Image, ImageDraw

    rng = random.Random(4)
    root = d / "seg"
    (root / "masks").mkdir(parents=True, exist_ok=True)
    for i in range(48):
        x, y, cls = rng.randint(10, 60), rng.randint(10, 60), 1 + i % 2
        kind, fill = ("rect", (230, 200, 30)) if cls == 1 else ("ellipse", (30, 200, 230))
        _img(root / "images" / f"{i}.png", 128, (90, 90, 90), rng, [(kind, (x, y, x + 50, y + 50), fill)])
        mask = Image.new("L", (128, 128), 0)
        draw = ImageDraw.Draw(mask)
        (draw.rectangle if kind == "rect" else draw.ellipse)((x, y, x + 50, y + 50), fill=cls)
        mask.save(root / "masks" / f"{i}.png")
    (root / "classes.txt").write_text("background\nsquare\ndisc\n")
    return root


def data_pairs(d: Path) -> Path:
    topics = ["refund policy", "shipping cost", "opening hours", "password reset", "invoice request", "warranty",
              "gift cards", "store locations", "order tracking", "account deletion"]
    rows = ["anchor,positive"]
    for i in range(200):
        t = topics[i % len(topics)]
        rows.append(f"\"How do I ask about {t}? ({i})\",\"Answer about {t}: see the {t} page, item {i % 7}.\"")
    p = d / "pairs.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def data_audio(d: Path) -> Path:
    import numpy as np
    import soundfile as sf

    root = d / "audio"
    root.mkdir(parents=True, exist_ok=True)
    words = ["one", "two", "three", "four", "five", "six"]
    rows = ["audio,text"]
    for i in range(24):
        t = np.linspace(0, 1.5, 24000, endpoint=False)
        sf.write(root / f"{i}.wav", 0.3 * np.sin(2 * np.pi * (220 + 40 * (i % 6)) * t), 16000)
        rows.append(f"{i}.wav,{words[i % 6]}")
    (root / "train.csv").write_text("\n".join(rows) + "\n")
    return root / "train.csv"


def data_style(d: Path) -> Path:
    rng = random.Random(5)
    root = d / "style"
    rows = ["file_name,text"]
    for i in range(8):
        _img(root / f"{i}.png", 128, (240, 120 + i * 10, 40), rng,
             [("ellipse", (30, 30, 98, 98), (250, 250, 220))])
        rows.append(f"{i}.png,a sks sunset with a pale disc")
    (root / "metadata.csv").write_text("\n".join(rows) + "\n")
    return root


def data_env(d: Path) -> Path:
    p = d / "cartpole.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("env_id: CartPole-v1\n")
    return p


# ---------------------------------------------------------------- checks


@dataclass
class Check:
    id: str
    pack: str
    recipe: str
    task_type: str
    make_data: Callable[[Path], Path]
    quick: dict[str, Any]
    full: dict[str, Any]
    metric: str  # the main metric (also what `expect` looks at)
    expect: Callable[[dict[str, float]], str | None]  # None = metrics look right, else the reason
    predict_row: Callable[[Path], dict[str, Any]]
    target: str | None = None
    needs: tuple[str, ...] = ()  # extra importable modules (checked before running)
    note: str = ""


def _at_least(metric: str, value: float):
    def f(m: dict[str, float]) -> str | None:
        v = m.get(metric)
        if v is None or not math.isfinite(v):
            return f"{metric} missing or not finite"
        return None if v >= value else f"{metric}={v:.3f} below {value}"
    return f


def _finite(metric: str):
    def f(m: dict[str, float]) -> str | None:
        v = m.get(metric)
        return None if v is not None and math.isfinite(v) else f"{metric} missing or not finite"
    return f


CHECKS: list[Check] = [
    Check("image-classification", "image-classification", "timm_finetune", "image-classification", data_images,
          {"model_name": "resnet18", "image_size": 128, "epochs": 5, "batch_size": 16},
          {"model_name": "convnext_tiny", "image_size": 224, "epochs": 3, "batch_size": 32},
          "accuracy", _at_least("accuracy", 0.9), lambda d: {"image": str(next((d / "imgs" / "red").glob("*.png")))}),
    Check("text-transformer", "text-classification", "transformer_finetune", "text-classification", data_reviews,
          {"epochs": 2, "batch_size": 16, "max_length": 64}, {"epochs": 3},
          "accuracy", _at_least("accuracy", 0.9), lambda d: {"text": "terrible, broke after a day"}, target="label"),
    Check("llm-qlora", "llm-finetune", "sft_lora", "llm-sft", data_chat,
          {"load_in_4bit": True, "epochs": 1, "batch_size": 4, "grad_accum": 1, "max_length": 128},
          {"model_name": "Qwen/Qwen2.5-1.5B-Instruct", "load_in_4bit": True, "epochs": 2, "max_length": 256},
          "val_loss", _finite("val_loss"), lambda d: {"prompt": "What is 3 plus 4?", "max_new_tokens": 16},
          needs=("bitsandbytes",), note="QLoRA 4-bit path (bitsandbytes)"),
    Check("llm-dpo", "llm-preference", "dpo_lora", "llm-preference", data_prefs,
          {"load_in_4bit": True, "epochs": 1, "batch_size": 2, "grad_accum": 1, "max_length": 128, "max_prompt_length": 64},
          {"model_name": "Qwen/Qwen2.5-1.5B-Instruct", "load_in_4bit": True, "epochs": 1, "max_length": 256},
          "reward_accuracy", _finite("reward_accuracy"), lambda d: {"prompt": "Reply politely about order 7.", "max_new_tokens": 24},
          needs=("bitsandbytes",)),
    Check("object-detection", "object-detection", "torchvision_detector", "object-detection", data_boxes,
          {"epochs": 6, "batch_size": 4, "min_size": 320, "max_size": 320},
          {"model_name": "fasterrcnn_resnet50_fpn_v2", "epochs": 10},
          "map50", _at_least("map50", 0.3), lambda d: {"image": str(d / "boxes" / "images" / "0.png")}),
    Check("image-segmentation", "image-segmentation", "torchvision_segmenter", "image-segmentation", data_masks,
          {"image_size": 128, "epochs": 8, "batch_size": 8},
          {"model_name": "deeplabv3_resnet50", "image_size": 320, "epochs": 15},
          "miou", _at_least("miou", 0.5), lambda d: {"image": str(d / "seg" / "images" / "0.png")}),
    Check("embedding", "embedding-finetune", "contrastive_pairs", "text-embedding", data_pairs,
          {"epochs": 1, "batch_size": 16}, {"epochs": 3},
          "recall_at_1", _at_least("recall_at_1", 0.5), lambda d: {"text": "How do I get a refund?", "compare": "refund policy answer"}),
    Check("whisper", "speech-recognition", "whisper_finetune", "speech-to-text", data_audio,
          {"model_name": "openai/whisper-tiny", "language": "en", "epochs": 1, "batch_size": 8, "grad_accum": 1},
          {"model_name": "openai/whisper-small", "language": "en", "epochs": 2},
          "wer", _finite("wer"), lambda d: {"audio": str(d / "audio" / "0.wav")},
          note="synthetic tones: checks the CUDA/bf16 training path, not transcription quality"),
    Check("diffusion-lora", "diffusion-lora", "sd_lora", "text-to-image", data_style,
          {"model_name": "hf-internal-testing/tiny-sd-pipe", "resolution": 64, "epochs": 2, "grad_accum": 1,
           "num_inference_steps": 4},
          {"resolution": 512, "max_steps": 100, "epochs": 50, "gradient_checkpointing": True},
          "val_loss", _finite("val_loss"), lambda d: {"prompt": "a sks sunset", "num_inference_steps": 4, "seed": 1},
          note="quick tier uses a tiny pipeline to exercise fp16/bf16 + GradScaler; full tier trains SD 1.5"),
    Check("rl-ppo", "reinforcement-learning", "sb3", "reinforcement-learning", data_env,
          {"total_timesteps": 20000, "eval_every_steps": 5000, "n_steps": 512}, {"total_timesteps": 100000},
          "mean_reward", _at_least("mean_reward", 50), lambda d: {"observation": [0.0, 0.0, 0.0, 0.0]},
          needs=("stable_baselines3", "gymnasium")),
]


def by_id(ids: list[str] | None) -> list[Check]:
    if not ids:
        return list(CHECKS)
    known = {c.id: c for c in CHECKS}
    missing = [i for i in ids if i not in known]
    if missing:
        raise ValueError(f"unknown check(s) {missing}; choose from {sorted(known)}")
    return [known[i] for i in ids]


# ---------------------------------------------------------------- GPU memory sampler


class VramSampler:
    """Peak GPU memory used on all devices while a run is active (NVML; 0 when unavailable)."""

    def __init__(self, interval: float = 0.25):
        self.interval, self.peak_gb, self.baseline_gb = interval, 0.0, 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:  # noqa: BLE001 — no NVML: report 0 instead of failing the check
            self._nvml, self._handles = None, []

    def _used_gb(self) -> float:
        return sum(self._nvml.nvmlDeviceGetMemoryInfo(h).used for h in self._handles) / 1024 ** 3 if self._handles else 0.0

    def __enter__(self) -> "VramSampler":
        self.baseline_gb = self._used_gb()
        if self._handles:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_gb = max(self.peak_gb, self._used_gb() - self.baseline_gb)

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


# ---------------------------------------------------------------- runner


@dataclass
class CheckResult:
    id: str
    pack: str
    recipe: str
    verdict: str  # pass | fail | skip | error
    reason: str
    device: str | None = None  # what the engine actually ran on
    seconds: float = 0.0
    peak_vram_gb: float = 0.0
    declared_min_vram_gb: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    prediction: str = ""
    run_id: str | None = None
    log_tail: list[str] = field(default_factory=list)


def _install(pack: str, packs_dir: Path | None) -> None:
    from autoforge.packs import installer
    from autoforge.packs.errors import PackExists

    try:
        installer.install(str(packs_dir / pack) if packs_dir else pack)
    except PackExists:
        pass


def run_check(check: Check, tier: str, device: str, work: Path, packs_dir: Path | None,
              timeout: float, hp_override: dict[str, Any] | None = None) -> CheckResult:
    from autoforge.engine import runner
    from autoforge.packs import installer
    from autoforge.registry import models as reg
    from autoforge.schemas.core import TaskSpec, TrainingPlan

    res = CheckResult(check.id, check.pack, check.recipe, "error", "")
    missing = [m for m in check.needs if importlib.util.find_spec(m) is None]
    if missing:
        res.verdict, res.reason = "skip", f"missing Python packages: {missing}"
        return res
    started = time.monotonic()
    try:
        _install(check.pack, packs_dir)
        recipe = installer.get_installed(check.pack)[1].recipe(check.recipe)
        res.declared_min_vram_gb = recipe.requires.min_vram_gb if recipe else None
        data = check.make_data(work / check.id)
        hp = dict(check.quick if tier == "quick" else check.full) | (hp_override or {})
        plan = TrainingPlan(task=TaskSpec(task_type=check.task_type, input_modality="x", output_modality="y",
                                          target=check.target, metric=check.metric), data_path=str(data), pack=check.pack,
                            recipe=check.recipe, device=device, hyperparameters=hp)
        with VramSampler() as vram:
            store = runner.create_run(plan, actor="ai:gpu-check")
            res.run_id = store.run_id
            runner.start(store, actor="ai:gpu-check")
            view = runner.wait(store, timeout)
        res.peak_vram_gb = round(vram.peak_gb, 2)
        res.log_tail = store.logs_tail(30)
        res.device = view.device
        if view.status.value != "succeeded":
            res.verdict, res.reason = "fail", f"run {view.status.value}: {(view.error or '').strip()[-300:]}"
            return res
        res.metrics = {k: float(v) for k, v in view.metrics.items()}
        problem = check.expect(res.metrics)
        name = f"gpucheck-{check.id}"
        reg.register(store.run_id, name)
        pred = reg.predict(name, [check.predict_row(work / check.id)])["predictions"][0]
        res.prediction = json.dumps(pred, ensure_ascii=False, default=str)[:200]
        if device == "cuda" and view.device != "cuda":
            problem = problem or f"asked for cuda but the run used {view.device}"
        elif device == "cuda" and vram._handles and res.peak_vram_gb < 0.05:
            problem = problem or "no GPU memory was used: the trainer may have run on the CPU"
        res.verdict, res.reason = ("fail", problem) if problem else ("pass", "trained on the GPU, metrics look right, predict works")
    except Exception as e:  # noqa: BLE001 — report and continue with the next check
        res.verdict, res.reason = "error", f"{type(e).__name__}: {e}"
    finally:
        res.seconds = round(time.monotonic() - started, 1)
    return res


def run_checks(checks: list[Check], tier: str = "quick", device: str = "cuda", packs_dir: Path | None = None,
               timeout: float = 3600, hp_override: dict[str, dict[str, Any]] | None = None,
               on_progress: Callable[[CheckResult], None] | None = None) -> dict[str, Any]:
    from autoforge.hardware.detect import detect_hardware

    hw = detect_hardware()
    out_dir = config.home() / "evals" / f"gpu-{time.strftime('%Y%m%d-%H%M%S')}"
    results: list[CheckResult] = []
    old = os.environ.get("AUTOFORGE_HOME")
    with tempfile.TemporaryDirectory(prefix="autoforge-gpucheck-") as tmp:
        os.environ["AUTOFORGE_HOME"] = str(Path(tmp) / "home")
        try:
            for check in checks:
                r = run_check(check, tier, device, Path(tmp) / "data", packs_dir, timeout,
                              (hp_override or {}).get(check.id))
                results.append(r)
                if on_progress:
                    on_progress(r)
        finally:
            if old is None:
                os.environ.pop("AUTOFORGE_HOME", None)
            else:
                os.environ["AUTOFORGE_HOME"] = old

    counts = {v: sum(r.verdict == v for r in results) for v in ("pass", "fail", "skip", "error")}
    report = {"tier": tier, "device": device, "gpus": [g.model_dump() for g in hw.gpus],
              "cuda": hw.cuda_version, "torch": hw.torch_version,
              "checks": [asdict(r) for r in results], "counts": counts, "total": len(results),
              "all_passed": counts["pass"] == len(results) - counts["skip"] and counts["pass"] > 0,
              "report_dir": str(out_dir)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(markdown(report), encoding="utf-8")
    return report


def markdown(report: dict[str, Any]) -> str:
    mark = {"pass": "✅", "fail": "❌", "skip": "⏭️", "error": "⚠️"}
    gpus = ", ".join(f"{g['name']} ({g['vram_total_gb']} GB)" for g in report["gpus"]) or "none"
    c = report["counts"]
    lines = [f"# GPU pack check — {report['tier']} tier", "",
             f"GPUs: {gpus}. Device: {report['device']}. {c['pass']} passed, {c['fail']} failed, "
             f"{c['error']} errors, {c['skip']} skipped.", "",
             "| check | result | time | peak VRAM | declared min | metrics | why |", "|---|---|---|---|---|---|---|"]
    for r in report["checks"]:
        metrics = ", ".join(f"{k}={v:.3g}" for k, v in list(r["metrics"].items())[:3])
        declared = f"{r['declared_min_vram_gb']} GB" if r["declared_min_vram_gb"] else "—"
        lines.append(f"| {r['id']} | {mark[r['verdict']]} {r['verdict']} | {r['seconds']:.0f} s | {r['peak_vram_gb']} GB | "
                     f"{declared} | {metrics} | {r['reason'].replace('|', '/')[:160]} |")
    return "\n".join(lines) + "\n"
