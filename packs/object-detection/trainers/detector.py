"""Fine-tune a torchvision Faster R-CNN to draw boxes around objects.

data_path/ (a folder) holds the images plus ONE annotation file:
  annotations.csv   columns image,xmin,ymin,xmax,ymax,label (pixel coordinates, image path relative to the
                    folder, one row per box; an image without objects may appear once with empty box fields)
  annotations.json  COCO: {"images": [{id, file_name}], "annotations": [{image_id, bbox: [x, y, w, h],
                    category_id}], "categories": [{id, name}]}
Or data_path/train/ and data_path/val/, each laid out as above; otherwise images are split by `val_fraction`.
AutoForge controls: check_stop per batch, live learning_rate (check_updates; linear warm-up over the first
epoch, at most 200 steps), checkpoint per epoch resumed by `run retry`, train_loss per step, map50 / precision / recall per epoch (best epoch kept).
"""

from __future__ import annotations

import base64
import csv
import io
import json
import math
import random
from pathlib import Path

MODELS = ("fasterrcnn_mobilenet_v3_large_fpn", "fasterrcnn_resnet50_fpn_v2")


# ---------- data ----------

def _num(v, where, col):
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: column '{col}' must be a number, got {v!r}") from None


def _read_csv(root: Path) -> list[dict]:
    path = root / "annotations.csv"
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        need = ["image", "xmin", "ymin", "xmax", "ymax", "label"]
        missing = [c for c in need if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}; expected columns {need}")
        items: dict[str, dict] = {}
        for i, row in enumerate(reader, start=2):
            name = (row["image"] or "").strip()
            if not name:
                raise ValueError(f"{path} line {i}: empty 'image' value")
            item = items.setdefault(name, {"path": root / name, "boxes": [], "labels": []})
            fields = [(row[c] or "").strip() for c in ("xmin", "ymin", "xmax", "ymax", "label")]
            if not any(fields):
                continue  # image with no objects
            if not all(fields):
                raise ValueError(f"{path} line {i}: give all of xmin,ymin,xmax,ymax,label or leave all empty")
            box = [_num(v, f"{path} line {i}", c) for v, c in zip(fields[:4], ("xmin", "ymin", "xmax", "ymax"))]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"{path} line {i}: need xmax > xmin and ymax > ymin (pixel coordinates), got {box}")
            item["boxes"].append(box)
            item["labels"].append(fields[4])
    return list(items.values())


def _read_coco(root: Path) -> list[dict]:
    path = root / "annotations.json"
    try:
        coco = json.loads(path.read_text(encoding="utf-8"))
        cats = {c["id"]: str(c["name"]) for c in coco["categories"]}
        images = {im["id"]: {"path": root / im["file_name"], "boxes": [], "labels": []} for im in coco["images"]}
        anns = coco.get("annotations", [])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path}: not a COCO file (need images[id,file_name], annotations, categories[id,name]): {e}") from None
    for a in anns:
        if a.get("iscrowd"):
            continue
        if a.get("image_id") not in images or a.get("category_id") not in cats:
            raise ValueError(f"{path}: annotation {a.get('id')} refers to unknown image_id or category_id")
        x, y, w, h = (float(v) for v in a["bbox"])
        if w <= 0 or h <= 0:
            continue
        images[a["image_id"]]["boxes"].append([x, y, x + w, y + h])
        images[a["image_id"]]["labels"].append(cats[a["category_id"]])
    return list(images.values())


def _read_folder(root: Path) -> list[dict]:
    if (root / "annotations.csv").is_file():
        items = _read_csv(root)
    elif (root / "annotations.json").is_file():
        items = _read_coco(root)
    else:
        raise FileNotFoundError(f"no annotations.csv or annotations.json (COCO) found in {root}")
    for it in items:
        if not Path(it["path"]).is_file():
            raise FileNotFoundError(f"image listed in annotations not found: {it['path']}")
    if not items:
        raise ValueError(f"annotations in {root} list no images")
    return items


def _split(root: Path, frac: float, seed: int):
    if not root.is_dir():
        raise FileNotFoundError(f"data_path must be a folder with images + annotations.csv/json, got {root}")
    if (root / "train").is_dir() and (root / "val").is_dir():
        return _read_folder(root / "train"), _read_folder(root / "val")
    items = _read_folder(root)
    random.Random(seed).shuffle(items)
    k = max(1, int(round(len(items) * frac))) if len(items) > 1 else 0
    return items[k:], items[:k]


def _load_image(src: str):
    from PIL import Image

    if src.startswith("data:") or (len(src) > 260 and not Path(src).exists()):
        raw = base64.b64decode(src.split(",", 1)[-1])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(src).convert("RGB")


def _sample(item, label_ids, flip):
    import torch
    import torchvision.transforms.functional as TF

    img = TF.to_tensor(_load_image(str(item["path"])))
    w = img.shape[-1]
    boxes = torch.tensor(item["boxes"], dtype=torch.float32).reshape(-1, 4)
    if flip:
        img = img.flip(-1)
        boxes = torch.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], 1) if len(boxes) else boxes
    labels = torch.tensor([label_ids[c] for c in item["labels"]], dtype=torch.int64)
    return img, {"boxes": boxes, "labels": labels}


def _batches(items, label_ids, bs, train, seed=0):
    rng = random.Random(seed)
    for i in range(0, len(items), bs):
        chunk = [_sample(it, label_ids, train and rng.random() < 0.5) for it in items[i:i + bs]]
        yield [c[0] for c in chunk], [c[1] for c in chunk]


# ---------- model ----------

def _build(name, num_classes, pretrained, min_size, max_size):
    import torchvision.models.detection as det
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if name not in MODELS:
        raise ValueError(f"model_name must be one of {list(MODELS)}, got {name!r}")
    kw = {"min_size": int(min_size), "max_size": int(max_size), "box_score_thresh": 0.01}
    if pretrained:
        model = getattr(det, name)(weights="DEFAULT", **kw)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    else:
        model = getattr(det, name)(weights=None, weights_backbone=None, num_classes=num_classes + 1, **kw)
    return model


def _match_frozen_bn(model, state):
    """COCO-pretrained backbones use FrozenBatchNorm2d (no num_batches_tracked); a model built without
    weights uses BatchNorm2d. Swap layers so a fine-tuned state dict loads into the weight-free skeleton."""
    import torch
    from torchvision.ops.misc import FrozenBatchNorm2d

    for name, mod in list(model.named_modules()):
        if isinstance(mod, torch.nn.BatchNorm2d) and f"{name}.weight" in state \
                and f"{name}.num_batches_tracked" not in state:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], FrozenBatchNorm2d(mod.num_features, eps=mod.eps))
    return model


# ---------- metrics ----------

def _iou(box, boxes):
    import numpy as np

    if len(boxes) == 0:
        return np.zeros(0)
    ix1, iy1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    ix2, iy2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def _match(dets, gts, cls, thr=0.5):
    """dets/gts: per-image lists of (boxes[N,4], labels[N], scores[N]). Returns (scores, tp flags, n_gt) for cls."""
    import numpy as np

    scores, tps, n_gt = [], [], 0
    for (db, dl, ds), (gb, gl, _) in zip(dets, gts):
        g = gb[gl == cls]
        n_gt += len(g)
        used = np.zeros(len(g), bool)
        m = dl == cls
        b, s = db[m], ds[m]
        for j in np.argsort(-s):
            ious = _iou(b[j], g)
            k = int(np.argmax(ious)) if len(ious) else -1
            hit = k >= 0 and ious[k] >= thr and not used[k]
            if hit:
                used[k] = True
            scores.append(float(s[j]))
            tps.append(hit)
    return np.array(scores), np.array(tps, bool), n_gt


def evaluate(dets, gts, num_classes, score_threshold):
    """VOC-style mAP@0.5 (all-point interpolation) over classes with ground truth, plus
    precision/recall of detections with score >= score_threshold."""
    import numpy as np

    aps, tp_t, det_t, gt_t = [], 0, 0, 0
    for cls in range(1, num_classes + 1):
        scores, tps, n_gt = _match(dets, gts, cls)
        keep = scores >= score_threshold
        tp_t += int(tps[keep].sum())
        det_t += int(keep.sum())
        gt_t += n_gt
        if n_gt == 0:
            continue
        order = np.argsort(-scores, kind="stable")
        tp = np.cumsum(tps[order])
        fp = np.cumsum(~tps[order])
        rec = tp / n_gt
        prec = tp / np.maximum(tp + fp, 1e-9)
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.concatenate([[0.0], prec, [0.0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        aps.append(float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])))
    return {"map50": float(np.mean(aps)) if aps else 0.0,
            "precision": tp_t / det_t if det_t else 0.0,
            "recall": tp_t / gt_t if gt_t else 0.0}


def _run_eval(model, items, label_ids, bs, device, num_classes, thr):
    import numpy as np
    import torch

    dets, gts = [], []
    model.eval()
    with torch.no_grad():
        for imgs, targets in _batches(items, label_ids, bs, train=False):
            for out, t in zip(model([im.to(device) for im in imgs]), targets):
                dets.append((out["boxes"].cpu().numpy(), out["labels"].cpu().numpy(), out["scores"].cpu().numpy()))
                gts.append((t["boxes"].numpy(), t["labels"].numpy(), None))
    if not gts:
        return {"map50": 0.0, "precision": 0.0, "recall": 0.0}
    return evaluate(dets, gts, num_classes, thr)


# ---------- train ----------

def train(ctx) -> dict:
    import torch

    hp = ctx.hyperparameters
    random.seed(ctx.seed)
    torch.manual_seed(ctx.seed)
    device = "cuda" if ctx.device == "cuda" and torch.cuda.is_available() else "cpu"
    train_items, val_items = _split(Path(ctx.data_path), float(hp.get("val_fraction", 0.2)), ctx.seed)
    classes = sorted({c for it in train_items + val_items for c in it["labels"]})
    if not classes:
        raise ValueError(f"no boxes found in the annotations under {ctx.data_path}")
    label_ids = {c: i + 1 for i, c in enumerate(classes)}  # 0 = background
    name = hp.get("model_name", MODELS[0])
    min_size, max_size = int(hp.get("min_size", 512)), int(hp.get("max_size", 853))
    thr = float(hp.get("score_threshold", 0.5))
    model = _build(name, len(classes), bool(hp.get("pretrained", True)), min_size, max_size)

    start_epoch, best, step = 0, None, 0
    state_file = ctx.latest_checkpoint()
    if state_file is not None and state_file.name == "state.json":
        state = json.loads(state_file.read_text())
        model.load_state_dict(torch.load(state["path"], map_location="cpu"))
        start_epoch, step = state["epoch"] + 1, state.get("step", 0)
        best = tuple(state["best"]) if state.get("best") else None
        ctx.log(f"resuming after epoch {state['epoch']}")
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=float(hp.get("learning_rate", 0.005)), momentum=0.9,
                          weight_decay=float(hp.get("weight_decay", 0.0005)))
    epochs, bs = int(hp.get("epochs", 10)), int(hp.get("batch_size", 4))
    base_lr = float(hp.get("learning_rate", 0.005))
    warmup = min(200, max(1, math.ceil(len(train_items) / bs)))
    flip = bool(hp.get("hflip", True))
    n_boxes = sum(len(it["boxes"]) for it in train_items)
    ctx.log(f"{name} on {device}: {len(train_items)} train / {len(val_items)} val images, "
            f"{n_boxes} train boxes, classes {classes}")

    for epoch in range(start_epoch, epochs):
        model.train()
        order = list(train_items)
        random.Random(ctx.seed + epoch).shuffle(order)
        total, nb = 0.0, 0
        for imgs, targets in _batches(order, label_ids, bs, train=flip, seed=ctx.seed + epoch):
            ctx.check_stop()
            changed = ctx.check_updates()
            if "learning_rate" in changed:
                base_lr = float(changed["learning_rate"])
            # linear warm-up over the first `warmup` steps (Faster R-CNN diverges without it)
            lr = base_lr * min(1.0, 0.001 + 0.999 * step / warmup) if step < warmup else base_lr
            for g in opt.param_groups:
                g["lr"] = lr
            losses = model([im.to(device) for im in imgs],
                           [{k: v.to(device) for k, v in t.items()} for t in targets])
            loss = sum(losses.values())
            lv = loss.item()
            if not math.isfinite(lv) and not any(len(t["boxes"]) for t in targets):
                ctx.log(f"skipped a batch of {len(imgs)} image(s) without boxes (no valid proposals)")
                continue
            if not math.isfinite(lv):
                parts = {k: round(v.item(), 4) for k, v in losses.items()}
                raise RuntimeError(f"training loss became {lv} {parts} at epoch {epoch}; lower learning_rate "
                                   f"(now {opt.param_groups[0]['lr']}) or check boxes are in pixel coordinates")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            ctx.log_metric("train_loss", lv, step=step)
            total, nb, step = total + lv, nb + 1, step + 1
        m = _run_eval(model, val_items, label_ids, bs, device, len(classes), thr)
        for k, v in m.items():
            ctx.log_metric(k, v, step=epoch)
        ctx.log(f"epoch {epoch}: train_loss={total / max(nb, 1):.4f} map50={m['map50']:.4f} "
                f"precision={m['precision']:.4f} recall={m['recall']:.4f}")
        ck = ctx.checkpoint_dir / f"epoch{epoch}.pt"
        torch.save(model.state_dict(), ck)
        if best is None or m["map50"] >= best[0]["map50"]:
            best = (m, str(ck), total / max(nb, 1))
        (ctx.checkpoint_dir / "state.json").write_text(json.dumps(
            {"epoch": epoch, "path": str(ck), "step": step, "best": list(best)}))

    out = Path(ctx.output_dir)
    if best:
        model.load_state_dict(torch.load(best[1], map_location="cpu"))
    metrics = dict(best[0]) if best else {"map50": 0.0, "precision": 0.0, "recall": 0.0}
    metrics["train_loss"] = best[2] if best else 0.0
    torch.save(model.state_dict(), out / "model.pt")
    (out / "model.json").write_text(json.dumps({"model_name": name, "classes": classes, "min_size": min_size,
                                                "max_size": max_size, "score_threshold": thr}))
    (out / "io.json").write_text(json.dumps({"features": [
        {"name": "image", "dtype": "image", "description": "image file path on the server, or base64 / data URL"},
        {"name": "score_threshold", "dtype": "numeric", "required": False,
         "description": f"minimum box score to return (default {thr})"}],
        "labels": classes}))
    return {"metrics": metrics, "model_path": str(out / "model.pt")}


# ---------- serving ----------

def load_model(model_dir):
    import torch

    meta = json.loads((Path(model_dir) / "model.json").read_text())
    model = _build(meta["model_name"], len(meta["classes"]), False, meta["min_size"], meta["max_size"])
    state = torch.load(Path(model_dir) / "model.pt", map_location="cpu")
    _match_frozen_bn(model, state).load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return {"model": model, "meta": meta, "device": device}


def predict(m, rows: list[dict]) -> list[dict]:
    import torch
    import torchvision.transforms.functional as TF

    classes, results = m["meta"]["classes"], []
    for r in rows:
        thr = r.get("score_threshold")
        thr = float(m["meta"]["score_threshold"] if thr is None else thr)
        x = TF.to_tensor(_load_image(str(r["image"]))).to(m["device"])
        with torch.no_grad():
            out = m["model"]([x])[0]
        found = [{"label": classes[int(lab) - 1], "score": round(float(s), 4),
                  "box": [round(float(v), 1) for v in b]}
                 for b, lab, s in zip(out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu())
                 if float(s) >= thr and 1 <= int(lab) <= len(classes)]
        results.append({"prediction": found, "count": len(found)})
    return results
