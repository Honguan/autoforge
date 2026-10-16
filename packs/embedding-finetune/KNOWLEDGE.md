# embedding-finetune — knowledge for AI planners

## When to use
- Search, RAG retrieval, FAQ / duplicate-question matching, recommendation by text similarity, where an
  off-the-shelf embedding model misses domain wording (legal, medical, product codes, internal jargon, Chinese slang).
- You have **pairs** of texts that should be close: (question, passage that answers it), (title, body),
  (query, clicked document), (paraphrase, paraphrase). 500+ pairs help; 5k+ pairs help a lot.

## When NOT to use
- You have class labels per text → `text-classification` (a classifier beats nearest-neighbour on labels).
- Fewer than ~200 pairs → use the base model unchanged; fine-tuning mostly overfits.
- You need generated answers → that is an LLM (`llm-finetune`) on top of retrieval, not an embedding.

## Data format
CSV or JSONL. Columns `anchor`, `positive`, optional `negative` (a hard negative: similar-looking but wrong).
Accepted aliases: `query`/`document`, `text1`/`text2`, `sentence1`/`sentence2`, `question`/`answer`; `hard_negative`.
```
anchor,positive,negative
how do I reset my password,"Go to Settings > Account > Reset password ...","To change your username, open ..."
退貨要多久,"退貨申請核准後 7 個工作天內退款",
```
No `target` column is needed. `val_fraction` of the pairs are held out for retrieval metrics.

## Recipe
`contrastive_pairs`: mean pooling (or `pooling: cls`) + L2 normalisation, MultipleNegativesRanking loss: each anchor
must pick its own positive among all positives (and hard negatives) in the batch; scores = cosine / `temperature`.
Default base `sentence-transformers/all-MiniLM-L6-v2` (384-dim, English, fast on CPU). Chinese / multilingual:
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `BAAI/bge-small-zh-v1.5` (use `pooling: cls`),
`intfloat/multilingual-e5-small`. Any BERT-like AutoModel works.

## Metrics
- On val pairs, each anchor retrieves its positive among **all** val positives: `recall_at_1` (HPO metric,
  **maximize**), `recall_at_5`, `mrr`. `baseline_recall_at_1` / `baseline_recall_at_5` / `baseline_mrr` are the
  base model's scores before training — always report the gain over the baseline.
- `train_loss` per step. Recall falls naturally as the val set grows (more distractors), so compare runs on the
  same data and `val_fraction`.

## Tuning guide
- `learning_rate` 1e-5–5e-5 (2e-5 default); above 1e-4 often destroys the pretrained space (recall below baseline).
- `batch_size` is the number of negatives: bigger is better (32–128 on GPU, 16–32 on CPU).
- `temperature` 0.02–0.1 (0.05 default); lower = sharper, can be unstable on noisy pairs.
- `epochs` 1–3 for 10k+ pairs, 3–5 for 1k pairs. `warmup_ratio` 0.1. `max_length` 128 for queries/short passages,
  256–512 for long passages (slower, more memory).
- Hard negatives (`negative` column) typically add 2–10 points of recall@1; mine them with the base model
  (top hits that are not the answer).

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| `need an anchor column ... found columns [...]` | column names not recognised | rename to `anchor`,`positive` |
| recall_at_1 below baseline | learning rate too high / too many epochs | `learning_rate: 1e-5`, `epochs: 1–2` |
| recall stuck, loss ≈ log(batch_size) | positives unrelated to anchors, or duplicates in batch | check pairs; deduplicate positives |
| CUDA out of memory | batch × max_length too large | lower `max_length` or `batch_size` |
| download error | no internet to huggingface.co | pre-download the model, set `model_name` to the local path |

## Serving
- Predict row `{"text": "..."}` → `{"prediction": [floats], "dim": n}` (L2-normalised, so dot product = cosine).
- Optional `{"text": a, "compare": b}` → also `"similarity"` (cosine, -1..1).
- The model dir is a standard Hugging Face model + tokenizer + `pooling.json` (and sentence-transformers config
  files). `model_export(format="onnx")` gives a graph with inputs `input_ids`, `attention_mask` and output
  `sentence_embedding` (pooled, normalised); needs `pip install onnx onnxscript`.
