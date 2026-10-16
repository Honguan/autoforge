"""Fine-tune a timm model on a folder-per-class dataset.

data_path/
  cat/ img1.jpg ...        (or data_path/train/<class>/ and data_path/val/<class>/)
  dog/ ...
AutoForge controls: check_stop per batch, live learning_rate, checkpoint per epoch (resumed by retry),
train_loss / val_loss / val_accuracy curves. The backbone is frozen for `freeze_backbone_epochs`.
"""

from __future__ import annotations

import base64
import io
import json
import random
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}


def _scan(root: Path) -> list[tuple[Path, str]]:
    items = []
    for cls in sorted(p for p in root.iterdir() if p.is_dir()):
        items += [(f, cls.name) for f in sorted(cls.rglob("*")) if f.suffix.lower() in EXTS]
    return items


def _split(ctx, frac, seed):
    root = Path(ctx.data_path)
    if (root / "train").is_dir() and (root / "val").is_dir():
        return _scan(root / "train"), _scan(root / "val")
    items = _scan(root)
    if not items:
        raise FileNotFoundError(f"no images found in class folders under {root}")
    random.Random(seed).shuffle(items)
    by_class: dict[str, list] = {}
    for it in items:
        by_class.setdefault(it[1], []).append(it)
    train, val = [], []
    for cls_items in by_class.values():  # stratified split
        k = max(1, int(len(cls_items) * frac)) if len(cls_items) > 1 else 0
        val += cls_items[:k]
        train += cls_items[k:]
    return train, val


def _transforms(model, image_size, train):
    import timm.data

    cfg = timm.data.resolve_data_config({"input_size": (3, image_size, image_size)}, model=model)
    return timm.data.create_transform(**cfg, is_training=train)


def _load_image(src: str):
    from PIL import Image

    if src.startswith("data:") or (len(src) > 260 and not Path(src).exists()):
        raw = base64.b64decode(src.split(",", 1)[-1])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(src).convert("RGB")


def _batches(items, tf, label_ids, bs):
    import torch

    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        x = torch.stack([tf(_load_image(str(p))) for p, _ in chunk])
        y = torch.tensor([label_ids[c] for _, c in chunk])
        yield x, y


def train(ctx) -> dict:
    import timm
    import torch

    hp = ctx.hyperparameters
    torch.manual_seed(ctx.seed)
    device = "cuda" if ctx.device == "cuda" and torch.cuda.is_available() else "cpu"
    train_items, val_items = _split(ctx, float(hp.get("val_fraction", 0.2)), ctx.seed)
    classes = sorted({c for _, c in train_items + val_items})
    label_ids = {c: i for i, c in enumerate(classes)}
    name, size = hp.get("model_name", "resnet18"), int(hp.get("image_size", 224))
    model = timm.create_model(name, pretrained=bool(hp.get("pretrained", True)), num_classes=len(classes))
    start_epoch = 0
    state_file = ctx.latest_checkpoint()
    if state_file is not None and state_file.name == "state.json":
        state = json.loads(state_file.read_text())
        model.load_state_dict(torch.load(state["path"], map_location="cpu"))
        start_epoch = state["epoch"] + 1
        ctx.log(f"resuming after epoch {state['epoch']}")
    model.to(device)
    tf_train, tf_val = _transforms(model, size, True), _transforms(model, size, False)
    head = set(id(p) for p in model.get_classifier().parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp.get("learning_rate", 3e-4)),
                            weight_decay=float(hp.get("weight_decay", 1e-4)))
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs, bs, freeze = int(hp.get("epochs", 5)), int(hp.get("batch_size", 32)), int(hp.get("freeze_backbone_epochs", 1))
    ctx.log(f"{name} on {device}: {len(train_items)} train / {len(val_items)} val images, classes {classes}")

    step, best = 0, None
    for epoch in range(start_epoch, epochs):
        for p in model.parameters():
            p.requires_grad = epoch >= freeze or id(p) in head
        model.train()
        random.Random(ctx.seed + epoch).shuffle(train_items)
        for x, y in _batches(train_items, tf_train, label_ids, bs):
            ctx.check_stop()
            changed = ctx.check_updates()
            if "learning_rate" in changed:
                for g in opt.param_groups:
                    g["lr"] = float(changed["learning_rate"])
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ctx.log_metric("train_loss", loss.item(), step=step)
            step += 1
        model.eval()
        total, correct, vloss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in _batches(val_items, tf_val, label_ids, bs):
                logits = model(x.to(device))
                vloss += float(loss_fn(logits, y.to(device))) * len(y)
                correct += int((logits.argmax(-1).cpu() == y).sum())
                total += len(y)
        acc, vl = correct / max(total, 1), vloss / max(total, 1)
        ctx.log_metric("val_loss", vl, step=epoch)
        ctx.log_metric("val_accuracy", acc, step=epoch)
        ctx.log(f"epoch {epoch}: val_loss={vl:.4f} val_accuracy={acc:.4f}")
        ck = ctx.checkpoint_dir / f"epoch{epoch}.pt"
        torch.save(model.state_dict(), ck)
        (ctx.checkpoint_dir / "state.json").write_text(json.dumps({"epoch": epoch, "path": str(ck)}))
        if best is None or acc >= best[0]:
            best = (acc, vl, ck)

    out = Path(ctx.output_dir)
    if best:
        model.load_state_dict(torch.load(best[2], map_location="cpu"))
    acc, vl = (best[0], best[1]) if best else (0.0, 0.0)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "model.json").write_text(json.dumps({"model_name": name, "classes": classes, "image_size": size}))
    (out / "io.json").write_text(json.dumps({"features": [
        {"name": "image", "dtype": "image", "description": "image file path on the server, or base64 / data URL"}],
        "labels": classes}))
    return {"metrics": {"accuracy": acc, "val_loss": vl}, "model_path": str(out / "model.pt")}


def load_model(model_dir):
    import timm
    import torch

    meta = json.loads((Path(model_dir) / "model.json").read_text())
    model = timm.create_model(meta["model_name"], pretrained=False, num_classes=len(meta["classes"]))
    model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return {"model": model, "meta": meta, "tf": _transforms(model, meta["image_size"], False), "device": device}


def predict(m, rows: list[dict]) -> list[dict]:
    import torch

    x = torch.stack([m["tf"](_load_image(str(r["image"]))) for r in rows]).to(m["device"])
    with torch.no_grad():
        probs = torch.softmax(m["model"](x), -1).cpu().tolist()
    classes = m["meta"]["classes"]
    return [{"prediction": classes[max(range(len(p)), key=p.__getitem__)],
             "probabilities": {c: round(v, 6) for c, v in zip(classes, p)}} for p in probs]


def export_onnx(model_dir, out_path, features) -> str:
    import torch

    m = load_model(model_dir)
    size = m["meta"]["image_size"]
    torch.onnx.export(m["model"].cpu(), torch.randn(1, 3, size, size), str(out_path),
                      input_names=["image"], output_names=["logits"], dynamic_axes={"image": {0: "batch"}})
    return str(out_path)
