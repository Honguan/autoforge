"""Fine-tune a pretrained transformer for sequence classification.

A plain PyTorch loop (no HF Trainer) so AutoForge controls work: ctx.check_stop()
every step, ctx.check_updates() for live learning-rate changes, one checkpoint per
epoch (resumed by `run retry`), train_loss / val_loss / val_f1 curves for `run diagnose`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import load_split, scores, write_io  # noqa: E402


def _batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _evaluate(model, tok, texts, label_ids, device, batch_size, max_length):
    import torch

    model.eval()
    preds, total_loss, n = [], 0.0, 0
    with torch.no_grad():
        for idx in _batches(list(range(len(texts))), batch_size):
            enc = tok([texts[i] for i in idx], truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
            labels = torch.tensor([label_ids[i] for i in idx], device=device)
            out = model(**enc, labels=labels)
            total_loss += out.loss.item() * len(idx)
            n += len(idx)
            preds += out.logits.argmax(-1).tolist()
    model.train()
    return total_loss / max(n, 1), preds


def train(ctx) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    hp = ctx.hyperparameters
    torch.manual_seed(ctx.seed)
    device = "cuda" if ctx.device == "cuda" and torch.cuda.is_available() else "cpu"
    x_train, x_val, y_train, y_val = load_split(ctx, hp.get("text_column", "text"), ctx.seed)
    labels = sorted(set(y_train) | set(y_val))
    label2id = {lab: i for i, lab in enumerate(labels)}
    yt, yv = [label2id[y] for y in y_train], [label2id[y] for y in y_val]

    source = hp.get("model_name", "distilbert-base-multilingual-cased")
    ckpt = ctx.latest_checkpoint()
    start_epoch = 0
    if ckpt is not None and ckpt.name == "state.json":
        state = json.loads(ckpt.read_text())
        source, start_epoch = state["path"], state["epoch"] + 1
        ctx.log(f"resuming after epoch {state['epoch']} from {source}")
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id,
        ignore_mismatched_sizes=True).to(device)

    epochs, bs, max_len = int(hp.get("epochs", 3)), int(hp.get("batch_size", 16)), int(hp.get("max_length", 128))
    lr = float(hp.get("learning_rate", 3e-5))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(hp.get("weight_decay", 0.01)))
    steps_per_epoch = math.ceil(len(x_train) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(float(hp.get("warmup_ratio", 0.1)) * steps_per_epoch * epochs),
                                            steps_per_epoch * epochs)
    ctx.log(f"{source} on {device}: {len(x_train)} train / {len(x_val)} val, {len(labels)} labels, {epochs} epochs")

    gen = torch.Generator().manual_seed(ctx.seed)
    step, best = start_epoch * steps_per_epoch, None
    model.train()
    for epoch in range(start_epoch, epochs):
        order = torch.randperm(len(x_train), generator=gen).tolist()
        for idx in _batches(order, bs):
            ctx.check_stop()
            changed = ctx.check_updates()
            if "learning_rate" in changed:
                for g in opt.param_groups:
                    g["lr"] = float(changed["learning_rate"])
            enc = tok([x_train[i] for i in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
            out = model(**enc, labels=torch.tensor([yt[i] for i in idx], device=device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            if step % 10 == 0:
                ctx.log_metric("train_loss", out.loss.item(), step=step)
            step += 1
        val_loss, preds = _evaluate(model, tok, x_val, yv, device, bs, max_len)
        m = scores(yv, preds)
        ctx.log_metric("val_loss", val_loss, step=epoch)
        ctx.log_metric("val_f1_macro", m["f1_macro"], step=epoch)
        ctx.log(f"epoch {epoch}: val_loss={val_loss:.4f} f1_macro={m['f1_macro']:.4f}")
        ck = ctx.checkpoint_dir / f"epoch{epoch}"
        model.save_pretrained(ck)
        tok.save_pretrained(ck)
        (ctx.checkpoint_dir / "state.json").write_text(json.dumps({"epoch": epoch, "path": str(ck)}))
        if best is None or m["f1_macro"] >= best[0]["f1_macro"]:
            best = (m, ck)

    if best is None:  # resumed after the last epoch
        best = (scores(yv, _evaluate(model, tok, x_val, yv, device, bs, max_len)[1]), None)
    out_dir = Path(ctx.output_dir)
    final = AutoModelForSequenceClassification.from_pretrained(best[1]) if best[1] else model
    final.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    write_io(out_dir, labels)
    return {"metrics": best[0], "model_path": str(out_dir)}


def load_model(model_dir):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    return {"model": model, "tok": AutoTokenizer.from_pretrained(model_dir), "device": device}


def predict(m, rows: list[dict]) -> list[dict]:
    import torch

    texts = [str(r.get("text", "")) for r in rows]
    enc = m["tok"](texts, truncation=True, max_length=256, padding=True, return_tensors="pt").to(m["device"])
    with torch.no_grad():
        probs = torch.softmax(m["model"](**enc).logits, -1).cpu().tolist()
    id2label = m["model"].config.id2label
    return [{"prediction": id2label[max(range(len(p)), key=p.__getitem__)],
             "probabilities": {id2label[i]: round(x, 6) for i, x in enumerate(p)}} for p in probs]
