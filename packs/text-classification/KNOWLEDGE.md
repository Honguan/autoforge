# text-classification — knowledge for AI planners

## When to use
- Each example is a piece of text with one label: sentiment, intent, topic, spam, ticket routing, language.
- Data: a CSV with a text column (default name `text`, set `text_column` otherwise) and the label column as `task.target`.

## When NOT to use
- Multiple labels per text (multi-label) — not supported yet.
- Extracting spans/entities (NER) or generating text — use an LLM pack.
- Long documents (> 512 tokens) where the label depends on the end: truncate wisely or summarise first.

## Choosing a recipe
| Situation | Recipe |
|---|---|
| Always first, any hardware, any language | `tfidf_linear` |
| GPU available, ≥ 1000 labelled texts, baseline not good enough | `transformer_finetune` |
| Chinese / Japanese / Korean text | `tfidf_linear` with `analyzer: char_wb` (default), or a multilingual / Chinese model |

Train `tfidf_linear` first: seconds on CPU and often within a few points of a transformer on topic/spam tasks.
Only move to `transformer_finetune` if the gap matters.

## Data requirements
- At least ~20 examples per label for the baseline; 100+ per label for fine-tuning.
- Labels are read as strings. Labels with a single example cannot be stratified; merge or drop them.
- Remove duplicates between train and future test data, or scores will be inflated.

## Metrics
- `f1_macro` for imbalanced labels (most real data), else `accuracy`.
- `transformer_finetune` also logs `train_loss`, `val_loss` and `val_f1_macro` per epoch; the best epoch is kept.

## Tuning guide
- `tfidf_linear`: `C` 1–10 is usually best; `analyzer: word` + `ngram_max: 2` for space-separated languages
  when vocabulary matters; `char_wb` + `ngram_max: 3–5` is robust to typos and works without tokenisation.
- `transformer_finetune`: `learning_rate` 2e-5–5e-5; 2–4 epochs; if `val_loss` rises after epoch 1–2 it is
  overfitting — fewer epochs or more data. Lower `batch_size` (8) or `max_length` (64) on GPU OOM.
- Models: `distilbert-base-multilingual-cased` (default, 104 languages), `bert-base-chinese` (Chinese only,
  stronger there), `xlm-roberta-base` (stronger multilingual, ~2× slower).

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'text'` | text column named differently | set `text_column` |
| f1 near 1/num_labels | labels shuffled or text column wrong | check data_profile |
| CUDA out of memory | batch too large for VRAM | `batch_size: 8`, `max_length: 64` |
| perfect score | duplicates / label leaked into text | deduplicate, check text content |
