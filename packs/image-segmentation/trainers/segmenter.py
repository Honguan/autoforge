"""Fine-tune a torchvision semantic segmentation model (LR-ASPP or DeepLabV3) on image + mask pairs.

data_path/
  images/  a.jpg b.png ...         any RGB image format
  masks/   a.png b.png ...         same file stem; single-channel PNG, pixel value = class index (0 = background),
                                   `ignore_index` (default 255) = not labelled / ignored
  classes.txt                      optional, one class name per line (line i = class i); else class_0, class_1, ...
or data_path/train/{images,masks} + data_path/val/{images,masks}; otherwise split by `val_fraction`.
AutoForge controls: check_stop per batch, live learning_rate (check_updates), checkpoint per epoch resumed by
`run retry`, train_loss per step, val_loss / miou / pixel_accuracy / iou_<class> per epoch (best epoch kept).
"""

from __future__ import annotations

import base64
import io
import json
import math
import random
import re
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MODELS = ("lraspp_mobilenet_v3_large", "deeplabv3_mobilenet_v3_large", "deeplabv3_resnet50")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------- data ----------

def _pairs(root: Path) -> list[tuple[Path, Path]]:
    img_dir, mask_dir = root / "images", root / "masks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"expected folders {img_dir} and {mask_dir} (masks = single-channel PNG, "
                                f"same file stem as the image)")
    masks = {p.stem: p for p in sorted(mask_dir.iterdir()) if p.suffix.lower() in EXTS}
    pairs = []
    for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in EXTS):
        if img.stem not in masks:
            raise FileNotFoundError(f"no mask for image {img}: expected {mask_dir / (img.stem + '.png')}")
        pairs.append((img, masks[img.stem]))
    if not pairs:
        raise FileNotFoundError(f"no images found in {img_dir}")
    return pairs


def _read_mask(path: Path):
    import numpy as np
    from PIL import Image

    im = Image.open(path)
    if im.mode not in ("L", "P", "I", "I;16", "1"):
        raise ValueError(f"mask {path} has mode {im.mode}; masks must be single-channel PNGs whose pixel "
                         f"value is the class index (0 = background)")
    return np.array(im).astype(np.int64)


def _split(root: Path, frac: float, seed: int):
    if not root.is_dir():
        raise FileNotFoundError(f"data_path must be a folder with images/ and masks/, got {root}")
    if (root / "train").is_dir() and (root / "val").is_dir():
        return _pairs(root / "train"), _pairs(root / "val")
    items = _pairs(root)
    random.Random(seed).shuffle(items)
    k = max(1, int(round(len(items) * frac))) if len(items) > 1 else 0
    return items[k:], items[:k]


def _classes(root: Path, pairs, ignore_index: int) -> list[str]:
    """Class names from classes.txt (or class_0..class_{max}); validates every mask value."""
    import numpy as np

    names = None
    for f in (root / "classes.txt", root / "train" / "classes.txt"):
        if f.is_file():
            names = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
            break
    seen: dict[int, Path] = {}
    for _, mp in pairs:
        for v in np.unique(_read_mask(mp)):
            if int(v) != ignore_index:
                seen.setdefault(int(v), mp)
    if names is None:
        names = [f"class_{i}" for i in range(max(max(seen, default=0) + 1, 2))]
    for v, mp in sorted(seen.items()):
        if v < 0 or v >= len(names):
            raise ValueError(f"mask {mp} contains value {v} but there are only {len(names)} classes "
                             f"(0..{len(names) - 1}{' from classes.txt' if names[0] != 'class_0' else ''}); "
                             f"use ignore_index={ignore_index} for unlabelled pixels")
    return names


def _load_image(src: str):
    from PIL import Image

    if src.startswith("data:") or (len(src) > 260 and not Path(src).exists()):
        raw = base64.b64decode(src.split(",", 1)[-1])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(src).convert("RGB")


def _to_tensor(img, size):
    import torchvision.transforms.functional as TF

    img = img.resize((size, size), resample=2)  # bilinear
    return TF.normalize(TF.to_tensor(img), MEAN, STD)


def _batches(items, size, bs, train, seed=0):
    import numpy as np
    import torch
    from PIL import Image

    rng = random.Random(seed)
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        if train and len(chunk) == 1 and len(items) > 1:
            continue  # BatchNorm in the ASPP pooling branch cannot train on a batch of 1
        xs, ys = [], []
        for ip, mp in chunk:
            x = _to_tensor(_load_image(str(ip)), size)
            m = Image.fromarray(_read_mask(mp).astype(np.int32), mode="I").resize((size, size), resample=0)
            y = torch.from_numpy(np.array(m).astype(np.int64))
            if train and rng.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            xs.append(x)
            ys.append(y)
        yield torch.stack(xs), torch.stack(ys)


# ---------- model ----------

def _build(name, num_classes, pretrained):
    import torch
    import torchvision.models.segmentation as seg

    if name not in MODELS:
        raise ValueError(f"model_name must be one of {list(MODELS)}, got {name!r}")
    if not pretrained:
        return getattr(seg, name)(weights=None, weights_backbone=None, num_classes=num_classes)
    model = getattr(seg, name)(weights="DEFAULT")  # COCO subset with the 21 VOC classes
    if name.startswith("lraspp"):
        for attr in ("low_classifier", "high_classifier"):
            old = getattr(model.classifier, attr)
            setattr(model.classifier, attr, torch.nn.Conv2d(old.in_channels, num_classes, 1))
    else:
        old = model.classifier[4]
        model.classifier[4] = torch.nn.Conv2d(old.in_channels, num_classes, 1)
        model.aux_classifier = None
    return model


# ---------- metrics ----------

def confusion(pred, target, num_classes, ignore_index):
    import numpy as np

    p, t = np.asarray(pred).ravel(), np.asarray(target).ravel()
    keep = (t != ignore_index) & (t >= 0) & (t < num_classes)
    return np.bincount(t[keep] * num_classes + p[keep], minlength=num_classes ** 2).reshape(num_classes, num_classes)


def scores(cm):
    """miou over classes present in prediction or target, pixel_accuracy, per-class IoU (None if absent)."""
    import numpy as np

    tp = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - tp
    ious = [float(tp[i] / union[i]) if union[i] > 0 else None for i in range(len(tp))]
    present = [v for v in ious if v is not None]
    return {"miou": float(np.mean(present)) if present else 0.0,
            "pixel_accuracy": float(tp.sum() / cm.sum()) if cm.sum() else 0.0, "per_class": ious}


def _run_eval(model, items, size, bs, device, n_cls, ignore_index, loss_fn):
    import numpy as np
    import torch

    model.eval()
    cm, vloss, n = np.zeros((n_cls, n_cls), np.int64), 0.0, 0
    with torch.no_grad():
        for x, y in _batches(items, size, bs, train=False):
            logits = model(x.to(device))["out"]
            vloss += float(loss_fn(logits, y.to(device))) * len(y)
            n += len(y)
            cm += confusion(logits.argmax(1).cpu().numpy(), y.numpy(), n_cls, ignore_index)
    s = scores(cm)
    s["val_loss"] = vloss / max(n, 1)
    return s


# ---------- train ----------

def train(ctx) -> dict:
    import torch

    hp = ctx.hyperparameters
    random.seed(ctx.seed)
    torch.manual_seed(ctx.seed)
    device = "cuda" if ctx.device == "cuda" and torch.cuda.is_available() else "cpu"
    root = Path(ctx.data_path)
    ignore_index = int(hp.get("ignore_index", 255))
    train_items, val_items = _split(root, float(hp.get("val_fraction", 0.2)), ctx.seed)
    classes = _classes(root, train_items + val_items, ignore_index)
    n_cls = len(classes)
    name, size = hp.get("model_name", MODELS[0]), int(hp.get("image_size", 320))
    model = _build(name, n_cls, bool(hp.get("pretrained", True)))

    start_epoch, best, step = 0, None, 0
    state_file = ctx.latest_checkpoint()
    if state_file is not None and state_file.name == "state.json":
        state = json.loads(state_file.read_text())
        model.load_state_dict(torch.load(state["path"], map_location="cpu"))
        start_epoch, step = state["epoch"] + 1, state.get("step", 0)
        best = tuple(state["best"]) if state.get("best") else None
        ctx.log(f"resuming after epoch {state['epoch']}")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp.get("learning_rate", 5e-4)),
                            weight_decay=float(hp.get("weight_decay", 1e-4)))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    epochs, bs, flip = int(hp.get("epochs", 15)), int(hp.get("batch_size", 8)), bool(hp.get("hflip", True))
    ctx.log(f"{name} on {device}: {len(train_items)} train / {len(val_items)} val images, "
            f"{n_cls} classes {classes}, image_size {size}")

    for epoch in range(start_epoch, epochs):
        model.train()
        order = list(train_items)
        random.Random(ctx.seed + epoch).shuffle(order)
        total, nb = 0.0, 0
        for x, y in _batches(order, size, bs, train=flip, seed=ctx.seed + epoch):
            ctx.check_stop()
            changed = ctx.check_updates()
            if "learning_rate" in changed:
                for g in opt.param_groups:
                    g["lr"] = float(changed["learning_rate"])
            loss = loss_fn(model(x.to(device))["out"], y.to(device))
            lv = loss.item()
            if not math.isfinite(lv):
                raise RuntimeError(f"training loss became {lv} at epoch {epoch}; lower learning_rate "
                                   f"(now {opt.param_groups[0]['lr']}) or check that masks hold class indices")
            opt.zero_grad()
            loss.backward()
            opt.step()
            ctx.log_metric("train_loss", lv, step=step)
            total, nb, step = total + lv, nb + 1, step + 1
        s = _run_eval(model, val_items, size, bs, device, n_cls, ignore_index, loss_fn) if val_items else \
            {"miou": 0.0, "pixel_accuracy": 0.0, "val_loss": 0.0, "per_class": [None] * n_cls}
        for k in ("miou", "pixel_accuracy", "val_loss"):
            ctx.log_metric(k, s[k], step=epoch)
        for cname, v in zip(classes, s["per_class"]):
            if v is not None:
                ctx.log_metric("iou_" + re.sub(r"[^A-Za-z0-9_]+", "_", cname), v, step=epoch)
        ctx.log(f"epoch {epoch}: train_loss={total / max(nb, 1):.4f} val_loss={s['val_loss']:.4f} "
                f"miou={s['miou']:.4f} pixel_accuracy={s['pixel_accuracy']:.4f}")
        ck = ctx.checkpoint_dir / f"epoch{epoch}.pt"
        torch.save(model.state_dict(), ck)
        if best is None or s["miou"] >= best[0]["miou"]:
            best = ({k: s[k] for k in ("miou", "pixel_accuracy", "val_loss", "per_class")}, str(ck),
                    total / max(nb, 1))
        (ctx.checkpoint_dir / "state.json").write_text(json.dumps(
            {"epoch": epoch, "path": str(ck), "step": step, "best": list(best)}))

    out = Path(ctx.output_dir)
    if best:
        model.load_state_dict(torch.load(best[1], map_location="cpu"))
    b = best[0] if best else {"miou": 0.0, "pixel_accuracy": 0.0, "val_loss": 0.0, "per_class": [None] * n_cls}
    metrics = {"miou": b["miou"], "pixel_accuracy": b["pixel_accuracy"], "val_loss": b["val_loss"],
               "train_loss": best[2] if best else 0.0}
    torch.save(model.state_dict(), out / "model.pt")
    (out / "model.json").write_text(json.dumps({"model_name": name, "classes": classes, "image_size": size,
                                                "per_class_iou": dict(zip(classes, b["per_class"]))}))
    (out / "io.json").write_text(json.dumps({"features": [
        {"name": "image", "dtype": "image", "description": "image file path on the server, or base64 / data URL"}],
        "labels": classes}))
    return {"metrics": metrics, "model_path": str(out / "model.pt")}


# ---------- serving ----------

def load_model(model_dir):
    import torch

    meta = json.loads((Path(model_dir) / "model.json").read_text())
    model = _build(meta["model_name"], len(meta["classes"]), False)
    model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return {"model": model, "meta": meta, "device": device}


def predict(m, rows: list[dict]) -> list[dict]:
    import numpy as np
    import torch
    from PIL import Image

    classes, size, results = m["meta"]["classes"], m["meta"]["image_size"], []
    for r in rows:
        img = _load_image(str(r["image"]))
        w, h = img.size
        x = _to_tensor(img, size).unsqueeze(0).to(m["device"])
        with torch.no_grad():
            logits = m["model"](x)["out"]
            logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        mask = logits.argmax(1)[0].cpu().numpy()
        counts = np.bincount(mask.ravel(), minlength=len(classes))
        buf = io.BytesIO()
        Image.fromarray(mask.astype(np.uint8 if len(classes) <= 256 else np.int32)).save(buf, format="PNG")
        results.append({"prediction": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                        "class_fractions": {c: round(float(n) / mask.size, 6) for c, n in zip(classes, counts)}})
    return results


def export_onnx(model_dir, out_path, features) -> str:
    """ONNX graph: input `image` float [N,3,S,S] RGB in 0..1 (normalisation is inside the graph),
    output `logits` [N,num_classes,S,S]; S = image_size from training."""
    import torch

    m = load_model(model_dir)
    size = m["meta"]["image_size"]

    class Wrapped(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

        def forward(self, image):
            return self.net((image - self.mean) / self.std)["out"]

    torch.onnx.export(Wrapped(m["model"].cpu()).eval(), torch.rand(1, 3, size, size), str(out_path),
                      input_names=["image"], output_names=["logits"], dynamic_axes={"image": {0: "batch"}})
    return str(out_path)
