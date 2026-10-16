"""LoRA fine-tuning of a Stable Diffusion 1.x / 2.x UNet (diffusers + peft) for text-to-image.

data_path/ is a folder of images (.jpg .jpeg .png .webp .bmp) with captions from, in this order:
  1. data_path/metadata.csv with columns file_name,text (file_name relative to data_path), or
  2. a same-stem .txt next to each image (dog1.jpg + dog1.txt), or
  3. the hyperparameter `instance_prompt` (DreamBooth style, e.g. "a photo of sks dog") for images without one.
Images are resized (shorter side) and cropped to `resolution` (center or random crop), optionally flipped.
Only LoRA layers on the UNet attention projections (to_q, to_k, to_v, to_out.0) train; VAE and text encoder are
frozen. Loss: MSE of the noise prediction (epsilon or v_prediction, from the scheduler config) with DDPMScheduler.
AutoForge controls: check_stop per micro-batch, live learning_rate via check_updates, LoRA checkpoint per epoch
(resumed by `run retry`), train_loss per optimizer step, deterministic val_loss per epoch (fixed images, noise and
timesteps); the epoch with the lowest val_loss is kept. validation_prompt samples go to output/samples/.
"""

from __future__ import annotations

import base64
import csv
import contextlib
import io
import json
import random
import shutil
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
VAL_TIMESTEPS = 4  # fixed timesteps per validation image


def _scan(root: Path, instance_prompt: str | None) -> list[tuple[Path, str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"data folder not found: {root} (expected a folder of images)")
    meta = root / "metadata.csv"
    items = []
    if meta.is_file():
        with meta.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"file_name", "text"} <= set(reader.fieldnames):
                raise ValueError(f"{meta}: needs columns file_name,text (got {reader.fieldnames})")
            for n, row in enumerate(reader, 2):
                img = root / (row["file_name"] or "")
                if not row["file_name"] or not img.is_file():
                    raise FileNotFoundError(f"{meta} line {n}: image '{row['file_name']}' not found in {root}")
                text = (row["text"] or "").strip() or instance_prompt
                if not text:
                    raise ValueError(f"{meta} line {n}: empty caption and no instance_prompt set")
                items.append((img, text))
    else:
        for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS):
            txt = img.with_suffix(".txt")
            text = txt.read_text(encoding="utf-8").strip() if txt.is_file() else ""
            text = text or instance_prompt
            if not text:
                raise ValueError(f"{img.name}: no caption. Add metadata.csv (file_name,text), a {txt.name} "
                                 "file, or set the instance_prompt hyperparameter")
            items.append((img, text))
    if not items:
        raise FileNotFoundError(f"no images ({', '.join(sorted(EXTS))}) found in {root}")
    return items


def _image_tensor(path: Path, resolution: int, center_crop: bool, flip: bool, rng: random.Random):
    import numpy as np
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB")
    scale = resolution / min(img.size)
    img = img.resize((max(resolution, round(img.width * scale)), max(resolution, round(img.height * scale))),
                     Image.BICUBIC)
    if center_crop:
        left, top = (img.width - resolution) // 2, (img.height - resolution) // 2
    else:
        left, top = rng.randint(0, img.width - resolution), rng.randint(0, img.height - resolution)
    img = img.crop((left, top, left + resolution, top + resolution))
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
    return x.permute(2, 0, 1)


def _dtype_kw(dtype) -> dict:
    """diffusers >= 0.36 renamed from_pretrained(torch_dtype=...) to dtype=..."""
    import diffusers

    new_api = tuple(int(x) for x in diffusers.__version__.split(".")[:2] if x.isdigit()) >= (0, 36)
    return {"dtype": dtype} if new_api else {"torch_dtype": dtype}


def _lora_state(unet):
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    return convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))


def _save_lora(unet, path: Path):
    from diffusers import StableDiffusionPipeline

    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(path), unet_lora_layers=_lora_state(unet), safe_serialization=True,
        unet_lora_adapter_metadata=unet.peft_config["default"].to_dict())


def _load_lora(unet, path: Path):
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_unet_state_dict_to_peft
    from peft import set_peft_model_state_dict

    sd = StableDiffusionPipeline.lora_state_dict(str(path))[0]
    sd = convert_unet_state_dict_to_peft({k[len("unet."):]: v for k, v in sd.items() if k.startswith("unet.")})
    set_peft_model_state_dict(unet, sd, adapter_name="default")


def train(ctx) -> dict:
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    hp = ctx.hyperparameters
    random.seed(ctx.seed)
    torch.manual_seed(ctx.seed)
    cuda = ctx.device == "cuda" and torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    base = hp.get("model_name", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    res = int(hp.get("resolution", 512))
    if res % 8:
        raise ValueError(f"resolution must be a multiple of 8, got {res}")
    center_crop, random_flip = bool(hp.get("center_crop", True)), bool(hp.get("random_flip", True))
    items = _scan(Path(ctx.data_path), hp.get("instance_prompt") or None)

    rng = random.Random(ctx.seed)
    rng.shuffle(items)
    n_val = int(len(items) * float(hp.get("val_fraction", 0.1)))
    if len(items) < 5 or n_val < 1:
        train_items, val_items = items, items
        ctx.log(f"only {len(items)} images: val_loss uses the training images with fixed noise (not a held-out set)")
    else:
        train_items, val_items = items[n_val:], items[:n_val]

    if cuda:
        mp = str(hp.get("mixed_precision", "auto"))
        if mp == "auto":
            mp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        wdtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mp, torch.float32)
    else:
        wdtype = torch.float32  # autocast / half precision only on CUDA
    tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", dtype=wdtype).to(device).eval()
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", **_dtype_kw(wdtype)).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", **_dtype_kw(wdtype)).to(device)
    sched = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=int(hp.get("lora_r", 16)), lora_alpha=int(hp.get("lora_alpha", 16)),
                                init_lora_weights="gaussian", target_modules=TARGETS))
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    for p in lora_params:  # LoRA weights train in fp32 even when the base is half precision
        p.data = p.data.float()
    if hp.get("gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
    vae_scale = vae.config.scaling_factor
    pred_type = sched.config.prediction_type
    T = sched.config.num_train_timesteps

    emb_cache: dict[str, torch.Tensor] = {}

    def embed(captions):
        missing = [c for c in dict.fromkeys(captions) if c not in emb_cache]
        if missing:
            ids = tok(missing, padding="max_length", max_length=tok.model_max_length, truncation=True,
                      return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                for c, e in zip(missing, text_encoder(ids)[0]):
                    emb_cache[c] = e
        return torch.stack([emb_cache[c] for c in captions])

    def encode(x, mean=False):
        with torch.no_grad():
            dist = vae.encode(x.to(device, dtype=wdtype)).latent_dist
            lat = dist.mean if mean else dist.sample()
        return lat * vae_scale

    def target_for(latents, noise, t):
        if pred_type == "v_prediction":
            return sched.get_velocity(latents, noise, t)
        return noise

    def autocast():
        if cuda and wdtype != torch.float32:
            return torch.autocast("cuda", dtype=wdtype)
        return contextlib.nullcontext()

    # fixed validation set: latents (VAE mean), noise and timesteps drawn once from a seeded generator
    gen = torch.Generator().manual_seed(ctx.seed)
    val_batches = []
    bs = int(hp.get("batch_size", 1))
    for i in range(0, len(val_items), bs):
        chunk = val_items[i:i + bs]
        x = torch.stack([_image_tensor(p, res, True, False, rng) for p, _ in chunk])
        lat = encode(x, mean=True).repeat_interleave(VAL_TIMESTEPS, 0)
        t = torch.tensor([int((k + 0.5) * T / VAL_TIMESTEPS) for _ in chunk for k in range(VAL_TIMESTEPS)])
        noise = torch.randn(lat.shape, generator=gen).to(device, dtype=lat.dtype)
        caps = [c for _, c in chunk for _ in range(VAL_TIMESTEPS)]
        val_batches.append((lat, noise, t.to(device), caps))

    def val_loss():
        unet.eval()
        total, n = 0.0, 0
        with torch.no_grad(), autocast():
            for lat, noise, t, caps in val_batches:
                pred = unet(sched.add_noise(lat, noise, t), t, embed(caps)).sample
                total += float(F.mse_loss(pred.float(), target_for(lat, noise, t).float(), reduction="sum"))
                n += pred.numel()
        unet.train()
        return total / max(n, 1)

    state_file = ctx.latest_checkpoint()
    start_epoch, step, best = 0, 0, None
    if state_file is not None and state_file.name == "state.json":
        state = json.loads(state_file.read_text())
        _load_lora(unet, Path(state["path"]))
        start_epoch, step = state["epoch"] + 1, state.get("step", 0)
        if state.get("best"):
            best = (state["best"]["val_loss"], Path(state["best"]["path"]))
        ctx.log(f"resuming after epoch {state['epoch']} (step {step})")

    accum = int(hp.get("grad_accum", 1))
    epochs, max_steps = int(hp.get("epochs", 10)), int(hp.get("max_steps", 0) or 0)
    opt = torch.optim.AdamW(lora_params, lr=float(hp.get("learning_rate", 1e-4)), weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda") if cuda and wdtype == torch.float16 else None
    ctx.log(f"{base} on {device} ({str(wdtype).split('.')[-1]}): {len(train_items)} train / {len(val_items)} val "
            f"images at {res}px, {sum(p.numel() for p in lora_params):,} LoRA parameters, {pred_type} loss")

    unet.train()
    for epoch in range(start_epoch, epochs):
        if max_steps and step >= max_steps:
            break
        erng = random.Random(ctx.seed + epoch)
        order = list(train_items)
        erng.shuffle(order)
        n_micro, run_loss = 0, 0.0
        for i in range(0, len(order), bs):
            ctx.check_stop()
            changed = ctx.check_updates()
            if "learning_rate" in changed:
                for g in opt.param_groups:
                    g["lr"] = float(changed["learning_rate"])
            chunk = order[i:i + bs]
            x = torch.stack([_image_tensor(p, res, center_crop, random_flip and erng.random() < 0.5, erng)
                             for p, _ in chunk])
            lat = encode(x)
            noise = torch.randn_like(lat)
            t = torch.randint(0, T, (lat.shape[0],), device=device)
            with autocast():
                pred = unet(sched.add_noise(lat, noise, t), t, embed([c for _, c in chunk])).sample
            loss = F.mse_loss(pred.float(), target_for(lat, noise, t).float())
            (scaler.scale(loss / accum) if scaler else loss / accum).backward()
            n_micro += 1
            run_loss += loss.item()
            if n_micro % accum == 0 or i + bs >= len(order):
                if scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()
                ctx.log_metric("train_loss", run_loss / (n_micro % accum or accum), step=step)
                run_loss = 0.0
                step += 1
                if max_steps and step >= max_steps:
                    break
        vl = val_loss()
        ctx.log_metric("val_loss", vl, step=epoch)
        ctx.log(f"epoch {epoch}: val_loss={vl:.5f} (step {step})")
        ck = ctx.checkpoint_dir / f"epoch{epoch}"
        _save_lora(unet, ck)
        if best is None or vl <= best[0]:
            best = (vl, ck)
        for old in ctx.checkpoint_dir.glob("epoch*"):  # keep only the latest and the best epoch
            if old.is_dir() and old not in (ck, best[1]):
                shutil.rmtree(old, ignore_errors=True)
        (ctx.checkpoint_dir / "state.json").write_text(json.dumps({
            "epoch": epoch, "step": step, "path": str(ck),
            "best": {"val_loss": best[0], "path": str(best[1])}}))

    out = Path(ctx.output_dir)
    if best and best[1].is_dir():  # keep the epoch with the lowest validation loss
        vl = best[0]
        shutil.copy2(best[1] / "pytorch_lora_weights.safetensors", out / "pytorch_lora_weights.safetensors")
    else:  # nothing trained in this process
        vl = val_loss()
        _save_lora(unet, out)
    (out / "model.json").write_text(json.dumps({
        "model_name": base, "resolution": res, "prediction_type": pred_type,
        "num_inference_steps": int(hp.get("num_inference_steps", 30)),
        "guidance_scale": float(hp.get("guidance_scale", 7.5)), "lora_r": int(hp.get("lora_r", 16))}))
    (out / "io.json").write_text(json.dumps({"features": [
        {"name": "prompt", "dtype": "text", "description": "what to draw; reuse the training trigger words"},
        {"name": "negative_prompt", "dtype": "text", "required": False, "description": "what to avoid"},
        {"name": "num_inference_steps", "dtype": "numeric", "required": False,
         "description": f"denoising steps, default {int(hp.get('num_inference_steps', 30))}"},
        {"name": "seed", "dtype": "numeric", "required": False, "description": "for reproducible images"}]}))

    prompt = hp.get("validation_prompt")
    if prompt:
        del unet, vae, text_encoder, opt, emb_cache, val_batches
        if cuda:
            torch.cuda.empty_cache()
        ctx.check_stop()
        m = load_model(out)
        samples = out / "samples"
        samples.mkdir(exist_ok=True)
        seed = int(hp.get("seed", ctx.seed) or 0)
        for k in range(int(hp.get("num_validation_images", 2))):
            r = predict(m, [{"prompt": prompt, "seed": seed + k}])[0]
            (samples / f"sample_{k}.png").write_bytes(base64.b64decode(r["prediction"].split(",", 1)[1]))
        ctx.log(f"saved {k + 1} sample(s) of '{prompt}' to {samples}")
    return {"metrics": {"val_loss": vl, "steps": step}, "model_path": str(out)}


def load_model(model_dir):
    import torch
    from diffusers import StableDiffusionPipeline

    model_dir = Path(model_dir)
    if model_dir.is_file():
        model_dir = model_dir.parent
    info = json.loads((model_dir / "model.json").read_text())
    cuda = torch.cuda.is_available()
    pipe = StableDiffusionPipeline.from_pretrained(info["model_name"],
                                                   **_dtype_kw(torch.float16 if cuda else torch.float32))
    pipe.load_lora_weights(str(model_dir))
    pipe = pipe.to("cuda" if cuda else "cpu")
    pipe.set_progress_bar_config(disable=True)
    return {"pipe": pipe, "info": info}


def predict(m, rows: list[dict]) -> list[dict]:
    import torch

    pipe, info, out = m["pipe"], m["info"], []
    for row in rows:
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("each row needs a non-empty 'prompt' string")
        seed = row.get("seed")
        seed = int(seed) if seed is not None and seed != "" else random.randint(0, 2**31 - 1)
        with torch.no_grad():
            img = pipe(prompt, negative_prompt=row.get("negative_prompt") or None,
                       num_inference_steps=int(row.get("num_inference_steps") or info.get("num_inference_steps", 30)),
                       guidance_scale=float(row.get("guidance_scale") or info.get("guidance_scale", 7.5)),
                       height=info["resolution"], width=info["resolution"],
                       generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append({"prediction": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), "seed": seed})
    return out
