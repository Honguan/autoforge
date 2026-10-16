"""Direct Preference Optimization (DPO) of an open LLM with LoRA / QLoRA (PEFT) in a plain PyTorch loop.

Data (JSONL, one preference pair per line):
  {"prompt": "...", "chosen": "better answer", "rejected": "worse answer"}
  "prompt" may also be a chat: [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
  (it should end with the user's turn; the tokenizer's chat template adds the assistant header).
Loss: sigmoid DPO, -log σ(β·[(log π(y_w|x) - log π_ref(y_w|x)) - (log π(y_l|x) - log π_ref(y_l|x))]),
optionally label-smoothed (conservative DPO). Log-probs are summed over response tokens only.
The policy is the base model + a LoRA adapter; the reference is the same base with the adapter disabled, so
no second model copy is loaded. `init_adapter` starts from an existing LoRA adapter (e.g. an llm-finetune run's
output); the reference is still the plain base model.
AutoForge controls: check_stop per micro-batch, live learning_rate and beta via check_updates, adapter checkpoint
per epoch (resumed by `run retry`), train_loss per optimizer step, val_loss / reward_accuracy / reward_margin per
epoch; the epoch with the highest reward_accuracy (then lowest val_loss) is kept.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

FALLBACK_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


def _read_pairs(path: str) -> list[tuple[list[dict] | str, str, str]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"preference data not found: {path} (expected a JSONL file)")
    pairs = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p.name} line {n}: not valid JSON ({e.msg})") from None
        if not isinstance(row, dict) or not all(k in row for k in ("prompt", "chosen", "rejected")):
            raise ValueError(f"{p.name} line {n}: needs 'prompt', 'chosen' and 'rejected' "
                             f"(got keys {sorted(row) if isinstance(row, dict) else type(row).__name__})")
        prompt, chosen, rejected = row["prompt"], row["chosen"], row["rejected"]
        if isinstance(prompt, list):
            if not prompt or not all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt):
                raise ValueError(f"{p.name} line {n}: 'prompt' as a list must be chat messages "
                                 "[{\"role\": ..., \"content\": ...}, ...]")
        elif not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{p.name} line {n}: 'prompt' must be a non-empty string or a messages list")
        for key, val in (("chosen", chosen), ("rejected", rejected)):
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{p.name} line {n}: '{key}' must be a non-empty string")
        if chosen == rejected:
            raise ValueError(f"{p.name} line {n}: 'chosen' and 'rejected' are identical")
        pairs.append((prompt, chosen, rejected))
    if len(pairs) < 2:
        raise ValueError(f"{p.name}: need at least 2 preference pairs, found {len(pairs)}")
    return pairs


def _prompt_text(tok, prompt) -> str:
    msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text = "\n\n".join(m["content"] for m in msgs)
    return FALLBACK_TEMPLATE.format(prompt=text)


def _encode_pair(tok, prompt, chosen, rejected, max_length, max_prompt_length):
    """-> ((ids, labels) chosen, (ids, labels) rejected); the prompt is left-truncated, responses right-truncated."""
    p = tok(_prompt_text(tok, prompt), add_special_tokens=False)["input_ids"][-max_prompt_length:]
    out = []
    for resp in (chosen, rejected):
        r = (tok(resp, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])[:max_length - len(p)]
        out.append((p + r, [-100] * len(p) + r))
    return out[0], out[1]


def _collate(seqs, pad_id, device):
    import torch

    width = max(len(ids) for ids, _ in seqs)
    ids = torch.full((len(seqs), width), pad_id)
    labels = torch.full((len(seqs), width), -100)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, (x, y) in enumerate(seqs):
        ids[i, :len(x)], labels[i, :len(y)], mask[i, :len(x)] = torch.tensor(x), torch.tensor(y), 1
    return ids.to(device), mask.to(device), labels.to(device)


def _logps(model, batch, pad_id, device):
    """Summed log-prob of the response tokens of each sequence; batch = list of (ids, labels)."""
    import torch

    ids, mask, labels = _collate(batch, pad_id, device)
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    labels = labels[:, 1:]
    keep = labels != -100
    tok_lp = torch.gather(logits.log_softmax(-1), 2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (tok_lp * keep).sum(-1)


def _pair_logps(model, chunk, pad_id, device):
    """chunk = list of (chosen, rejected, ...) -> (logp_chosen, logp_rejected), one forward pass."""
    lp = _logps(model, [c[0] for c in chunk] + [c[1] for c in chunk], pad_id, device)
    return lp[:len(chunk)], lp[len(chunk):]


def _ref_logps(model, data, bs, pad_id, device):
    """Reference log-probs (adapter disabled) for every pair, computed once: the reference never changes."""
    import torch

    model.eval()
    out = []
    with torch.no_grad(), model.disable_adapter():
        for i in range(0, len(data), bs):
            c, r = _pair_logps(model, data[i:i + bs], pad_id, device)
            out += list(zip(c.tolist(), r.tolist()))
    model.train()
    return [(ch, rj, rc, rr) for (ch, rj), (rc, rr) in zip(data, out)]


def _dpo(pc, pr, rc, rr, beta, smoothing):
    import torch.nn.functional as F

    logits = beta * ((pc - rc) - (pr - rr))
    loss = -F.logsigmoid(logits) * (1 - smoothing) - F.logsigmoid(-logits) * smoothing
    return loss, beta * (pc - rc), beta * (pr - rr)


def _evaluate(model, data, bs, pad_id, device, beta, smoothing):
    import torch

    model.eval()
    loss, correct, margin, n = 0.0, 0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(data), bs):
            chunk = data[i:i + bs]
            pc, pr = _pair_logps(model, chunk, pad_id, device)
            rc = torch.tensor([c[2] for c in chunk], device=pc.device)
            rr = torch.tensor([c[3] for c in chunk], device=pc.device)
            l, cw, rw = _dpo(pc, pr, rc, rr, beta, smoothing)
            loss += float(l.sum())
            correct += int((cw > rw).sum())
            margin += float((cw - rw).sum())
            n += len(chunk)
    model.train()
    n = max(n, 1)
    return {"val_loss": loss / n, "reward_accuracy": correct / n, "reward_margin": margin / n}


def train(ctx) -> dict:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hp = ctx.hyperparameters
    random.seed(ctx.seed)
    torch.manual_seed(ctx.seed)
    cuda = ctx.device == "cuda" and torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    base = hp.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    max_len, max_prompt = int(hp.get("max_length", 1024)), int(hp.get("max_prompt_length", 512))
    if max_prompt >= max_len:
        raise ValueError(f"max_prompt_length ({max_prompt}) must be smaller than max_length ({max_len})")

    pairs = _read_pairs(ctx.data_path)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * float(hp.get("eval_fraction", 0.1))))
    val = [_encode_pair(tok, p, c, r, max_len, max_prompt) for p, c, r in pairs[:n_val]]
    train_set = [_encode_pair(tok, p, c, r, max_len, max_prompt) for p, c, r in pairs[n_val:]]

    kwargs: dict = {}
    if hp.get("load_in_4bit"):
        if not cuda:
            raise RuntimeError("load_in_4bit (QLoRA) needs an NVIDIA GPU; set load_in_4bit: false on CPU")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if hp.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.to(device)

    state_file = ctx.latest_checkpoint()
    start_epoch, best = 0, None
    init_adapter = hp.get("init_adapter")
    if state_file is not None and state_file.name == "state.json":
        state = json.loads(state_file.read_text())
        model = PeftModel.from_pretrained(model, state["path"], is_trainable=True)
        start_epoch = state["epoch"] + 1
        if state.get("best"):
            best = (tuple(state["best"]["key"]), state["best"]["metrics"], Path(state["best"]["path"]))
        ctx.log(f"resuming after epoch {state['epoch']}")
    elif init_adapter:
        if not (Path(init_adapter) / "adapter_config.json").is_file():
            raise FileNotFoundError(f"init_adapter {init_adapter}: no adapter_config.json there "
                                    "(point it at the output folder of an llm-finetune run)")
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        ctx.log(f"starting from adapter {init_adapter} (reference = base model without adapter)")
    else:
        model = get_peft_model(model, LoraConfig(
            r=int(hp.get("lora_r", 16)), lora_alpha=int(hp.get("lora_alpha", 32)),
            lora_dropout=float(hp.get("lora_dropout", 0.05)), target_modules="all-linear", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bs, accum, epochs = int(hp.get("batch_size", 2)), int(hp.get("grad_accum", 8)), int(hp.get("epochs", 1))
    ctx.log(f"DPO {base} on {device}{' (4-bit)' if hp.get('load_in_4bit') else ''}: {len(train_set)} train / "
            f"{len(val)} val pairs, {trainable:,} trainable parameters, beta={hp.get('beta', 0.1)}")

    pad = tok.pad_token_id
    train_set = _ref_logps(model, train_set, bs, pad, device)
    val = _ref_logps(model, val, bs, pad, device)
    smoothing = float(hp.get("label_smoothing", 0.0))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(hp.get("learning_rate", 5e-5)))
    model.train()
    step = 0
    for epoch in range(start_epoch, epochs):
        random.Random(ctx.seed + epoch).shuffle(train_set)
        for i in range(0, len(train_set), bs):
            ctx.check_stop()
            changed = ctx.check_updates()
            if "learning_rate" in changed:
                for g in opt.param_groups:
                    g["lr"] = float(changed["learning_rate"])
            chunk = train_set[i:i + bs]
            pc, pr = _pair_logps(model, chunk, pad, device)
            rc = torch.tensor([c[2] for c in chunk], device=pc.device)
            rr = torch.tensor([c[3] for c in chunk], device=pc.device)
            loss = _dpo(pc, pr, rc, rr, float(hp.get("beta", 0.1)), smoothing)[0].mean()
            (loss / accum).backward()
            if (i // bs + 1) % accum == 0 or i + bs >= len(train_set):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                ctx.log_metric("train_loss", loss.item(), step=step)
                step += 1
        m = _evaluate(model, val, bs, pad, device, float(hp.get("beta", 0.1)), smoothing)
        for k, v in m.items():
            ctx.log_metric(k, v, step=epoch)
        ctx.log(f"epoch {epoch}: val_loss={m['val_loss']:.4f} reward_accuracy={m['reward_accuracy']:.3f} "
                f"reward_margin={m['reward_margin']:.4f}")
        ck = ctx.checkpoint_dir / f"epoch{epoch}"
        model.save_pretrained(ck)
        key = (m["reward_accuracy"], -m["val_loss"])
        if best is None or key >= best[0]:
            best = (key, m, ck)
        (ctx.checkpoint_dir / "state.json").write_text(json.dumps({
            "epoch": epoch, "path": str(ck),
            "best": {"key": list(best[0]), "metrics": best[1], "path": str(best[2])}}))

    out = Path(ctx.output_dir)
    if best and best[2].is_dir():  # keep the epoch with the highest reward accuracy
        metrics = best[1]
        shutil.copytree(best[2], out, dirs_exist_ok=True)
    else:  # nothing trained in this process and no saved best
        metrics = _evaluate(model, val, bs, pad, device, float(hp.get("beta", 0.1)), smoothing)
        model.save_pretrained(out)
    tok.save_pretrained(out)  # adapter only; the base model is referenced in adapter_config.json
    (out / "io.json").write_text(json.dumps({"features": [
        {"name": "prompt", "dtype": "text", "description": "instruction or question for the model"},
        {"name": "max_new_tokens", "dtype": "numeric", "required": False, "description": "default 256"}]}))
    return {"metrics": dict(metrics), "model_path": str(out)}


def load_model(model_dir):
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoPeftModelForCausalLM.from_pretrained(model_dir).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    return {"model": model, "tok": tok, "device": device}


def predict(m, rows: list[dict]) -> list[dict]:
    import torch

    tok, model, out = m["tok"], m["model"], []
    for row in rows:
        prompt = row.get("prompt", "")
        prompt = prompt if isinstance(prompt, list) else str(prompt)
        enc = tok(_prompt_text(tok, prompt), return_tensors="pt", add_special_tokens=False).to(m["device"])
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=int(row.get("max_new_tokens") or 256), do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        out.append({"prediction": tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)})
    return out


def export_merged(model_dir, out_path, features) -> str:
    """Merge the LoRA adapter into the base model and save full weights (safetensors) to a directory."""
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    merged = AutoPeftModelForCausalLM.from_pretrained(model_dir).merge_and_unload()
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_dir).save_pretrained(out)
    return str(out)
