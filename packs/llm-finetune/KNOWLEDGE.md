# llm-finetune — knowledge for AI planners

## When to use
- Teach an open LLM a **behaviour**: answer format, tone, domain vocabulary, a narrow task (extraction into JSON,
  ticket replies, classification with explanations), or a language style.
- Data: JSONL, one example per line: `{"prompt": ..., "response": ...}` or `{"messages": [...]}`
  (system/user/assistant; the last message is the assistant answer that is trained).

## When NOT to use
- Plain classification with fixed labels → `text-classification` (faster, cheaper, more accurate).
- Facts that change often or a large document base → retrieval (RAG), not fine-tuning.
- Fewer than ~100 good examples → prompt engineering / few-shot first.

## Choosing settings by GPU (NVIDIA)
| VRAM | Model size | Settings |
|---|---|---|
| 6–8 GB | 0.5–1.5B (Qwen2.5-0.5B/1.5B-Instruct) | `load_in_4bit: false`, `batch_size: 4` |
| 8–12 GB | 7–8B (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct) | `load_in_4bit: true` (QLoRA), `batch_size: 1–2`, `grad_accum: 8–16` |
| 24 GB | 7–8B | `load_in_4bit: false` or QLoRA with `max_length: 2048` |
| 24 GB+ | 13–14B | QLoRA |

Measured (`autoforge eval gpu`, RTX 4060 Laptop 8 GB, Windows): Qwen2.5-0.5B-Instruct QLoRA, `max_length: 128`,
`batch_size: 4` peaked at 1.4 GB. Longer `max_length` grows activation memory roughly linearly.
QLoRA needs `pip install bitsandbytes`. Gated models (Llama) need `huggingface-cli login` first.

## Data requirements
- 200–2,000 high-quality examples beat 20,000 noisy ones. Responses must be exactly what you want back.
- Keep `max_length` above your longest example (tokens), or long answers are cut.
- Use the base model's chat format: the trainer applies the tokenizer's chat template automatically.

## Metrics
- `val_loss` (lower is better) and `val_perplexity`; the epoch with the lowest `val_loss` is kept.
- Loss alone does not prove quality: after training, call model_predict on a few held-out prompts and read them.

## Tuning guide
- `learning_rate`: 1e-4–2e-4 for LoRA; lower (5e-5) if `train_loss` spikes or `val_loss` diverges.
- `lora_r`: 8–16 for style/format, 32–64 for harder new skills. `lora_alpha` ≈ 2 × `lora_r`.
- `epochs`: 1–3. If `val_loss` rises after epoch 1 → overfitting: fewer epochs or more data.
- Effective batch = `batch_size × grad_accum` ≈ 16–32.

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| CUDA out of memory | model/batch too big | QLoRA, `batch_size: 1`, raise `grad_accum`, lower `max_length` |
| `load_in_4bit ... needs an NVIDIA GPU` | QLoRA on CPU | `load_in_4bit: false` or use a GPU |
| repeats the prompt / never stops | no EOS learned / wrong template | check examples; the trainer appends EOS |
| 401 / gated repo | model needs licence acceptance | accept on huggingface.co, `huggingface-cli login` |

## Export
- `model_export(format="merged")` merges LoRA into the base weights (safetensors) for vLLM / llama.cpp conversion.
