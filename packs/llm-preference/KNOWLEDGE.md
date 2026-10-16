# llm-preference — knowledge for AI planners

## When to use
- You have pairs of answers to the same prompt where one is **better** (`chosen`) and one **worse** (`rejected`):
  human ratings, thumbs up/down logs, A/B picks, or a stronger model judging two outputs.
- Goal: shift an instruct model's behaviour (tone, conciseness, refusing/complying, format discipline, fewer
  hallucinated details) without writing perfect target answers. Best as the step **after** `llm-finetune` (SFT).

## When NOT to use
- You only have good answers, no rejected ones → `llm-finetune` (SFT).
- Fixed labels → `text-classification`. Knowledge that changes often → retrieval (RAG).
- Fewer than ~200 pairs, or `chosen` and `rejected` barely differ → little signal; collect more pairs first.

## Data format
JSONL, one pair per line. `prompt` is a string or a chat-message list ending with the user turn:
```
{"prompt": "Summarise: the meeting moved to Friday.", "chosen": "Meeting is now on Friday.", "rejected": "Sure! Here is a summary: ..."}
{"prompt": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}], "chosen": "Hello!", "rejected": "Greetings, ..."}
```
The tokenizer's chat template is applied to the prompt; EOS is appended to both answers. Long prompts are cut from
the left to `max_prompt_length`; answers are cut so prompt + answer ≤ `max_length`.

## Recipe
`dpo_lora` — sigmoid DPO (optionally label-smoothed) on a LoRA adapter over all linear layers; QLoRA with
`load_in_4bit: true`. The **reference model is the base model with the adapter disabled** (no second copy in
memory; reference log-probs are computed once before training). `init_adapter: <path>` starts from an existing LoRA
adapter, e.g. the `output/` folder of an llm-finetune run (same `model_name`!). The reference stays the plain base,
so DPO also keeps the SFT gains as part of the "policy"; use a smaller `beta`/fewer epochs if outputs drift.
GPU sizing is the same as llm-finetune (6–8 GB: 0.5–1.5B; 8–12 GB: 7–8B with QLoRA), but each step runs two
sequences per pair, so halve `batch_size` and raise `grad_accum`.

## Metrics
- `reward_accuracy` (val, **maximize for HPO**): share of held-out pairs where the implicit reward
  β·(log π − log π_ref) of `chosen` beats `rejected`. 0.5 = no preference learned.
- `reward_margin` (val): mean reward gap; grows as the model separates the answers.
- `val_loss` (val DPO loss; starts at ln 2 ≈ 0.693 from a fresh adapter) and `train_loss` per optimizer step.
- The epoch with the highest `reward_accuracy` (ties → lowest `val_loss`) is kept. Always read a few generations.

## Tuning guide
- `beta` 0.05–0.5 (default 0.1): lower lets the model move further from the base; higher keeps it close.
- `learning_rate` 5e-6–1e-4 for LoRA (default 5e-5). Loss stuck at 0.693 → raise; collapse/gibberish → lower.
- `epochs` 1–3: DPO overfits fast; if `reward_accuracy` falls while `train_loss` → 0, stop at epoch 1.
- `label_smoothing` 0.1–0.2 when preferences are noisy (crowd or model-judged labels).
- Effective batch `batch_size × grad_accum` ≈ 16–64.

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| CUDA out of memory | two sequences per pair | QLoRA, `batch_size: 1`, raise `grad_accum`, lower `max_length` |
| `val_loss` stays ≈ 0.693, accuracy ≈ 0.5 | lr too low or pairs not distinguishable | raise `learning_rate`, check data |
| outputs become long/repetitive or gibberish | over-optimisation (reward hacking) | raise `beta`, fewer epochs, lower lr |
| `needs 'prompt', 'chosen' and 'rejected'` | wrong JSONL keys | rename fields |
| `init_adapter ... no adapter_config.json` | path is not a LoRA adapter folder | use the run's `output/` folder |
| `load_in_4bit ... needs an NVIDIA GPU` | QLoRA on CPU | `load_in_4bit: false` or use a GPU |

## Serving
Input row `{"prompt": "...", "max_new_tokens": 256}` (`prompt` may be a messages list) → `{"prediction": "<answer>"}`
(greedy decoding). `model_export(format="merged")` merges the adapter into the base weights (safetensors).
