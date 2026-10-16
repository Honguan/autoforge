"""Supervised fine-tuning with LoRA / QLoRA (PEFT) in a plain PyTorch loop.

Data (JSONL, one example per line), either:
  {"prompt": "...", "response": "..."}
  {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}   (last message = assistant target)
Only the response tokens are trained on (prompt tokens are masked with -100).
AutoForge controls: check_stop per micro-batch, live learning_rate via check_updates,
adapter checkpoint per epoch (resumed by `run retry`), train_loss / val_loss / val_perplexity curves.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

FALLBACK_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


def _read_examples(path: str) -> list[tuple[list[dict] | str, str]]:
    examples = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" in row:
            msgs = row["messages"]
            if not msgs or msgs[-1].get("role") != "assistant":
                raise ValueError(f"line {n}: the last message must be the assistant's answer")
            examples.append((msgs[:-1], msgs[-1]["content"]))
        elif "prompt" in row and "response" in row:
            examples.append((row["prompt"], row["response"]))
        else:
            raise ValueError(f"line {n}: needs 'prompt' + 'response' or 'messages'")
    if len(examples) < 2:
        raise ValueError("need at least 2 examples")
    return examples


def _prompt_text(tok, prompt) -> str:
    msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text = "\n\n".join(m["content"] for m in msgs)
    return FALLBACK_TEMPLATE.format(prompt=text)


def _encode(tok, prompt, response, max_length):
    p = tok(_prompt_text(tok, prompt), add_special_tokens=False)["input_ids"]
    r = tok(response, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p + r)[:max_length]
    labels = ([-100] * len(p) + r)[:max_length]
    return ids, labels


def _collate(batch, pad_id, device):
    import torch

    width = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), width), pad_id)
    labels = torch.full((len(batch), width), -100)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)], labels[i, :len(y)], mask[i, :len(x)] = torch.tensor(x), torch.tensor(y), 1
    return ids.to(device), mask.to(device), labels.to(device)


def _val_loss(model, data, bs, pad_id, device):
    import torch

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(data), bs):
            ids, mask, labels = _collate(data[i:i + bs], pad_id, device)
            total += float(model(input_ids=ids, attention_mask=mask, labels=labels).loss) * len(data[i:i + bs])
            n += len(data[i:i + bs])
    model.train()
    return total / max(n, 1)


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

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    examples = _read_examples(ctx.data_path)
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * float(hp.get("eval_fraction", 0.1))))
    max_len = int(hp.get("max_length", 1024))
    val = [_encode(tok, p, r, max_len) for p, r in examples[:n_val]]
    train_set = [_encode(tok, p, r, max_len) for p, r in examples[n_val:]]

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
        model = prepare_model_for_kbit_training(model)
    else:
        model.to(device)

    state_file = ctx.latest_checkpoint()
    start_epoch = 0
    if state_file is not None and state_file.name == "state.json":
        state = json.loads(state_file.read_text())
        model = PeftModel.from_pretrained(model, state["path"], is_trainable=True)
        start_epoch = state["epoch"] + 1
        ctx.log(f"resuming after epoch {state['epoch']}")
    else:
        model = get_peft_model(model, LoraConfig(
            r=int(hp.get("lora_r", 16)), lora_alpha=int(hp.get("lora_alpha", 32)),
            lora_dropout=float(hp.get("lora_dropout", 0.05)), target_modules="all-linear", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ctx.log(f"{base} on {device}{' (4-bit)' if hp.get('load_in_4bit') else ''}: {len(train_set)} train / "
            f"{len(val)} val examples, {trainable:,} trainable parameters")

    bs, accum, epochs = int(hp.get("batch_size", 4)), int(hp.get("grad_accum", 4)), int(hp.get("epochs", 2))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(hp.get("learning_rate", 2e-4)))
    model.train()
    step, best = 0, None
    for epoch in range(start_epoch, epochs):
        random.shuffle(train_set)
        for i in range(0, len(train_set), bs):
            ctx.check_stop()
            changed = ctx.check_updates()
            if "learning_rate" in changed:
                for g in opt.param_groups:
                    g["lr"] = float(changed["learning_rate"])
            ids, mask, labels = _collate(train_set[i:i + bs], tok.pad_token_id, device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            (loss / accum).backward()
            if (i // bs + 1) % accum == 0 or i + bs >= len(train_set):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                ctx.log_metric("train_loss", loss.item(), step=step)
                step += 1
        vl = _val_loss(model, val, bs, tok.pad_token_id, device)
        ctx.log_metric("val_loss", vl, step=epoch)
        ctx.log(f"epoch {epoch}: val_loss={vl:.4f} perplexity={math.exp(min(vl, 50)):.2f}")
        ck = ctx.checkpoint_dir / f"epoch{epoch}"
        model.save_pretrained(ck)
        (ctx.checkpoint_dir / "state.json").write_text(json.dumps({"epoch": epoch, "path": str(ck)}))
        if best is None or vl <= best[0]:
            best = (vl, ck)

    out = Path(ctx.output_dir)
    if best:  # keep the epoch with the lowest validation loss
        vl = best[0]
        shutil.copytree(best[1], out, dirs_exist_ok=True)
    else:  # resumed after the last epoch: nothing new was trained
        vl = _val_loss(model, val, bs, tok.pad_token_id, device)
        model.save_pretrained(out)
    tok.save_pretrained(out)  # adapter only (small); the base model is referenced in adapter_config.json
    (out / "io.json").write_text(json.dumps({"features": [
        {"name": "prompt", "dtype": "text", "description": "instruction or question for the model"},
        {"name": "max_new_tokens", "dtype": "numeric", "required": False, "description": "default 256"}]}))
    return {"metrics": {"val_loss": vl, "val_perplexity": math.exp(min(vl, 50))}, "model_path": str(out)}


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
        enc = tok(_prompt_text(tok, str(row.get("prompt", ""))), return_tensors="pt", add_special_tokens=False).to(m["device"])
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
