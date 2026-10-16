"""Fine-tune a text embedding model on (anchor, positive[, negative]) pairs with in-batch negatives.

Data: CSV or JSONL (.jsonl / .json lines), one pair per row:
  anchor,positive[,negative]        e.g. a query and the passage that answers it
  Also accepted: query/document, text1/text2, sentence1/sentence2, question/answer; negative/hard_negative.
Plain transformers (AutoModel + AutoTokenizer): mean (or CLS) pooling over the attention mask, L2-normalised,
MultipleNegativesRanking loss = cross-entropy over cosine similarities / temperature, where every other
positive in the batch (and every hard negative) is a wrong candidate.
Validation: retrieve each val anchor's positive among all val positives -> recall_at_1 / recall_at_5 / mrr
(also measured before training as baseline_*).
AutoForge controls: check_stop per batch, live learning_rate via check_updates, checkpoint per epoch
(save_pretrained + checkpoints/state.json, resumed by `run retry`), the best epoch (recall_at_1) is kept.
Output: a standard Hugging Face model dir + tokenizer + pooling.json (+ sentence-transformers config files).
"""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

ANCHOR = ["anchor", "query", "text1", "sentence1", "question"]
POSITIVE = ["positive", "document", "text2", "sentence2", "answer", "passage"]
NEGATIVE = ["negative", "hard_negative"]


def _pick(cols, names):
    lower = {str(c).strip().lower(): c for c in cols}
    return next((lower[n] for n in names if n in lower), None)


def read_pairs(path: str) -> list[tuple[str, str, str | None]]:
    """Read (anchor, positive, negative-or-None) rows from a CSV or JSONL file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"data file not found: {p} (expected a CSV or JSONL file with anchor,positive columns)")
    if p.suffix.lower() in (".jsonl", ".json", ".ndjson"):
        rows = []
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as e:
                    raise ValueError(f"{p.name} line {n}: not valid JSON ({e})") from None
        cols = list(dict.fromkeys(k for r in rows for k in r))
    else:
        import pandas as pd

        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        rows, cols = df.to_dict("records"), list(df.columns)
    a, b, neg = _pick(cols, ANCHOR), _pick(cols, POSITIVE), _pick(cols, NEGATIVE)
    if a is None or b is None:
        raise ValueError(f"{p.name}: need an anchor column ({'/'.join(ANCHOR)}) and a positive column "
                         f"({'/'.join(POSITIVE)}); found columns {cols}")
    pairs = []
    for r in rows:
        x, y = str(r.get(a) or "").strip(), str(r.get(b) or "").strip()
        z = str(r.get(neg) or "").strip() if neg else ""
        if x and y:
            pairs.append((x, y, z or None))
    if len(pairs) < 4:
        raise ValueError(f"{p.name}: need at least 4 non-empty ({a}, {b}) pairs, found {len(pairs)}")
    return pairs


def _encode(model, tok, texts, pooling, max_length, device, bs=64, grad=False):
    import torch

    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.set_grad_enabled(grad):
            hidden = model(**enc).last_hidden_state
            if pooling == "cls":
                emb = hidden[:, 0]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            outs.append(torch.nn.functional.normalize(emb, dim=-1))
    return torch.cat(outs)


def _retrieval(model, tok, val, pooling, max_length, device) -> dict:
    """Each val anchor must find its own positive among all val positives (identical texts count as hits)."""
    import torch

    model.eval()
    with torch.no_grad():
        a = _encode(model, tok, [x for x, _, _ in val], pooling, max_length, device)
        p = _encode(model, tok, [y for _, y, _ in val], pooling, max_length, device)
        sims = (a @ p.T).cpu()
    texts = [y for _, y, _ in val]
    r1 = r5 = rr = 0.0
    for i in range(len(val)):
        wrong = torch.tensor([texts[j] != texts[i] for j in range(len(val))])
        rank = int(((sims[i] > sims[i, i]) & wrong).sum()) + 1
        r1, r5, rr = r1 + (rank == 1), r5 + (rank <= 5), rr + 1.0 / rank
    n = max(len(val), 1)
    return {"recall_at_1": r1 / n, "recall_at_5": r5 / n, "mrr": rr / n}


def _save(model, tok, out: Path, pooling: str, max_length: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    dim = int(model.config.hidden_size)
    (out / "pooling.json").write_text(json.dumps({"pooling": pooling, "normalize": True, "max_length": max_length,
                                                  "dim": dim}))
    # sentence-transformers layout, so SentenceTransformer(out) also loads it
    (out / "modules.json").write_text(json.dumps([
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}]))
    (out / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": max_length, "do_lower_case": False}))
    (out / "1_Pooling").mkdir(exist_ok=True)
    (out / "1_Pooling" / "config.json").write_text(json.dumps({
        "word_embedding_dimension": dim, "pooling_mode_cls_token": pooling == "cls",
        "pooling_mode_mean_tokens": pooling != "cls", "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False}))
    (out / "2_Normalize").mkdir(exist_ok=True)


def train(ctx) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    hp = ctx.hyperparameters
    random.seed(ctx.seed)
    torch.manual_seed(ctx.seed)
    device = "cuda" if ctx.device == "cuda" and torch.cuda.is_available() else "cpu"
    base = hp.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    pooling = str(hp.get("pooling", "mean")).lower()
    if pooling not in ("mean", "cls"):
        raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")
    max_len, bs = int(hp.get("max_length", 128)), int(hp.get("batch_size", 32))
    epochs, temp = int(hp.get("epochs", 3)), float(hp.get("temperature", 0.05))

    pairs = read_pairs(ctx.data_path)
    random.shuffle(pairs)
    n_val = min(len(pairs) - 2, max(2, int(len(pairs) * float(hp.get("val_fraction", 0.1)))))
    val, train_set = pairs[:n_val], pairs[n_val:]

    state_file = ctx.latest_checkpoint()
    state = json.loads(state_file.read_text()) if state_file is not None and state_file.name == "state.json" else None
    src = state["path"] if state else base
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModel.from_pretrained(src).to(device)
    start_epoch, best = 0, None
    if state:
        start_epoch, best = state["epoch"] + 1, state.get("best")
        ctx.log(f"resuming after epoch {state['epoch']}")
    n_neg = sum(1 for *_, z in train_set if z)
    ctx.log(f"{base} on {device}: {len(train_set)} train / {len(val)} val pairs ({n_neg} with hard negatives), "
            f"pooling={pooling}, batch_size={bs}")
    if bs < 8:
        ctx.log("note: batch_size < 8 gives few in-batch negatives; 32-128 works best")

    if start_epoch == 0:
        base_m = _retrieval(model, tok, val, pooling, max_len, device)
        for k, v in base_m.items():
            ctx.log_metric(f"baseline_{k}", v, step=0)
        ctx.log("before training: " + " ".join(f"{k}={v:.4f}" for k, v in base_m.items()))

    lr = float(hp.get("learning_rate", 2e-5))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    per_epoch = math.ceil(len(train_set) / bs)
    total = max(1, per_epoch * epochs)
    warm = int(total * float(hp.get("warmup_ratio", 0.1)))
    loss_fn = torch.nn.CrossEntropyLoss()
    step = start_epoch * per_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        random.Random(ctx.seed + epoch).shuffle(train_set)
        for i in range(0, len(train_set), bs):
            ctx.check_stop()
            changed = ctx.check_updates()
            if "learning_rate" in changed:
                lr = float(changed["learning_rate"])
            scale = (step + 1) / max(warm, 1) if step < warm else max(0.0, (total - step) / max(total - warm, 1))
            for g in opt.param_groups:
                g["lr"] = lr * scale
            batch = train_set[i:i + bs]
            if len(batch) < 2:
                continue  # a single pair has no in-batch negatives
            a = _encode(model, tok, [x for x, _, _ in batch], pooling, max_len, device, bs=len(batch), grad=True)
            cands = [y for _, y, _ in batch] + [z for *_, z in batch if z]
            c = _encode(model, tok, cands, pooling, max_len, device, bs=len(cands), grad=True)
            loss = loss_fn((a @ c.T) / temp, torch.arange(len(batch), device=device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ctx.log_metric("train_loss", loss.item(), step=step)
            step += 1
        m = _retrieval(model, tok, val, pooling, max_len, device)
        for k, v in m.items():
            ctx.log_metric(k, v, step=epoch)
        ctx.log(f"epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
        ck = ctx.checkpoint_dir / f"epoch{epoch}"
        _save(model, tok, ck, pooling, max_len)
        if best is None or m["recall_at_1"] >= best["metrics"]["recall_at_1"]:
            old, best = best, {"epoch": epoch, "path": str(ck), "metrics": m}
            if old and Path(old["path"]).is_dir():
                shutil.rmtree(old["path"], ignore_errors=True)
        (ctx.checkpoint_dir / "state.json").write_text(json.dumps({"epoch": epoch, "path": str(ck), "best": best}))
        prev = ctx.checkpoint_dir / f"epoch{epoch - 1}"  # keep only the latest and the best epoch on disk
        if epoch > 0 and prev.is_dir() and str(prev) != best["path"]:
            shutil.rmtree(prev, ignore_errors=True)

    out = Path(ctx.output_dir)
    if best and Path(best["path"]).is_dir():
        shutil.copytree(best["path"], out, dirs_exist_ok=True)
        metrics = best["metrics"]
        ctx.log(f"kept epoch {best['epoch']} (recall_at_1={metrics['recall_at_1']:.4f})")
    else:  # nothing trained (e.g. resumed after the last epoch, best dir gone)
        metrics = _retrieval(model, tok, val, pooling, max_len, device)
        _save(model, tok, out, pooling, max_len)
    (out / "io.json").write_text(json.dumps({"features": [
        {"name": "text", "dtype": "text", "description": "text to embed"},
        {"name": "compare", "dtype": "text", "required": False,
         "description": "optional second text; the result then also has its cosine 'similarity' to text"}]}))
    return {"metrics": dict(metrics), "model_path": str(out)}


def load_model(model_dir):
    import torch
    from transformers import AutoModel, AutoTokenizer

    cfg = json.loads((Path(model_dir) / "pooling.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_dir).to(device).eval()
    return {"model": model, "tok": AutoTokenizer.from_pretrained(model_dir), "cfg": cfg, "device": device}


def predict(m, rows: list[dict]) -> list[dict]:
    import torch

    cfg = m["cfg"]

    def enc(texts):
        with torch.no_grad():
            return _encode(m["model"], m["tok"], texts, cfg["pooling"], cfg["max_length"], m["device"]).cpu()

    for r in rows:
        if not isinstance(r.get("text"), str):
            raise ValueError("each row needs a 'text' string")
    emb = enc([r["text"] for r in rows])
    cmp_idx = [i for i, r in enumerate(rows) if r.get("compare") is not None]
    other = dict(zip(cmp_idx, enc([str(rows[i]["compare"]) for i in cmp_idx]))) if cmp_idx else {}
    out = []
    for i, e in enumerate(emb):
        res = {"prediction": [round(float(v), 6) for v in e], "dim": int(e.shape[0])}
        if i in other:
            res["similarity"] = round(float(e @ other[i]), 6)
        out.append(res)
    return out


def export_onnx(model_dir, out_path, features) -> str:
    """ONNX graph: inputs input_ids, attention_mask [batch, seq] -> sentence_embedding [batch, dim] (pooled, normalised)."""
    import torch

    m = load_model(model_dir)
    enc, pooling = m["model"].cpu(), m["cfg"]["pooling"]

    class Embedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc

        def forward(self, input_ids, attention_mask):
            hidden = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if pooling == "cls":
                emb = hidden[:, 0]
            else:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(emb, dim=-1)

    sample = m["tok"](["an example sentence", "another one"], padding=True, return_tensors="pt")
    torch.onnx.export(Embedder().eval(), (sample["input_ids"], sample["attention_mask"]), str(out_path),
                      input_names=["input_ids", "attention_mask"], output_names=["sentence_embedding"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                    "sentence_embedding": {0: "batch"}})
    return str(out_path)
