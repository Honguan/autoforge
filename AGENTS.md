# AGENTS.md — 給 AI 的使用說明

You are operating AutoForge, a tool that trains AI models on the user's machine.
Prefer the MCP server (`autoforge mcp`, tools listed in docs/MCP.md) when your client supports MCP;
the CLI below offers the same operations with `--json`.

## Rules
1. Always call `autoforge doctor --json` first to learn the hardware.
2. Every command accepts `--json` and returns `{"ok": bool, "data": ..., "error": {"code","message","hint"}}`.
   Branch on `error.code`, and follow `error.hint`.
3. Read data contracts with `autoforge schema <Name> --json` instead of guessing fields.
4. Privacy: if an error has code `PRIVACY_DENIED`, do not retry. Tell the user the dataset
   is sensitive and needs a local-model brain.
5. Training is asynchronous: `run start` returns a `run_id` at once; poll `run status <id>`.
6. Identify yourself: pass `--actor ai:<your-name>` (or set AUTOFORGE_ACTOR) so the audit log
   shows who did what. Humans and AI share the same controls.

## Data and privacy
- `autoforge data profile <path> --json` (MCP: `data_profile`): rows, columns, dtypes, missing values.
- Data is `internal` unless a human marked it. Cloud AI sees statistics only; `public` data adds
  sample rows; `sensitive` data is refused with PRIVACY_DENIED. Never try to change sensitivity.

## Finding and learning a training method
1. `autoforge pack search [keywords] --task <task-type> --fit --json`
   `--fit` keeps only recipes this machine can run. Each hit lists `runnable_recipes`.
2. `autoforge pack install <name> --json` installs the commit pinned in the index.
3. `autoforge pack info <name> --json` returns the manifest, recipes (defaults, search space)
   and `knowledge` (KNOWLEDGE.md). **Read `knowledge` before planning.**
4. If `missing_dependencies` is non-empty, tell the user to `pip install` them.

## Training (P3)
1. Write a plan file (schema: `autoforge schema TrainingPlan --json`). Paths are relative to the plan file.
   Example: `examples/churn_plan.yaml`.
2. `autoforge plan validate plan.yaml --json`: deterministic checks (pack/recipe/task match,
   data and target column exist, GPU/VRAM, missing packages). Fix every `problems` item.
3. `autoforge run start plan.yaml --json --actor ai:<name>` → `data.run_id`
4. Poll `autoforge run status <id> --json` (status: pending/running/paused/succeeded/failed/stopped).
   For progress use `autoforge run metrics <id> --json` (last/min/max/count/recent per metric);
   on failure read `autoforge run logs <id> --tail 50 --json`.
5. Control: `run pause | resume | stop <id>`; `run retry <id>` reruns the plan and resumes from checkpoints.
6. Custom trainers: set `custom_code_path: my_trainer.py[:function]` instead of pack+recipe.
   The code is copied into the run and committed to the run's git repo. Contract: docs/PACK_SPEC.md §5.

## Improving a model (P5)
- `run diagnose <id>` / MCP `run_diagnose`: findings (failure cause, divergence, plateau, overfitting,
  leakage) with suggestions and the tunable hyperparameters.
- `hpo start plan.yaml -m <metric> -n <trials>` / MCP `hpo_start` → `hpo_wait`: search; trial 0 is the
  plan's own hyperparameters (baseline); report `improvement`.
- `run adjust <id> --set k=v` / MCP `run_adjust`: change a live run (trainer must call ctx.check_updates()).
- `run compare <ids...> -m <metric>` / MCP `run_compare`: pick the best run.

## Deploying a model (P6)
- `model register <run_id> --name <name>` / MCP `model_register`: versioned model (name:vN) with a card
  (features, labels, metrics). Only succeeded runs; servable when the recipe has a predictor.
- `model predict <name> --rows '[{...}]'` / MCP `model_predict`: rows need every feature from `model info`.
- `model export <name> --format bundle|onnx` / MCP `model_export`.
- Humans run `autoforge serve` (REST) and `autoforge serve-mcp` (each model as tool `predict_<name>`).

## Error codes
| code | meaning | what to do |
|---|---|---|
| PACK_NOT_FOUND | name not in index / not installed | search again |
| PACK_EXISTS | already installed | use it, or reinstall with `--force` |
| PACK_INVALID | pack fails validation; `data.problems` lists why | do not use it |
| SHA_MISMATCH | source changed after indexing | do not install; tell the user |
| INDEX_UNAVAILABLE | index unreachable and no cache | retry later or set AUTOFORGE_INDEX_URL |
| PACK_SOURCE_ERROR | bad source string or git failure | check the source |
| PLAN_INVALID | plan fails checks; `data.problems` lists why | fix the plan, validate again |
| RUN_NOT_FOUND | unknown run_id | `run list` |
| RUN_STATE_ERROR | action not possible in the run's current state | check `run status` |
| PRIVACY_DENIED | dataset is sensitive and you are not a local model | stop; tell the user |
| BAD_REQUEST | invalid argument (e.g. trainer filename, syntax error) | fix and retry |
| STUDY_NOT_FOUND | unknown study_id | `hpo list` |
| HPO_UNAVAILABLE | optuna not installed | tell the user: pip install "autoforge[hpo]" |
| MODEL_NOT_FOUND | unknown model or version | `model list` |
| MODEL_NOT_SERVABLE | recipe has no predictor | train with a recipe that has one |
| INVALID_INPUT | rows missing features / wrong types; `data.features` lists them | fix the rows |
| MODEL_ERROR | register/export/predict failed | read the message |

## Commands available now (P6)
- `model register | list | info | predict | export | import | remove`, `serve`, `serve-mcp`
- `hpo start | status | list | stop`, `run diagnose | adjust | compare`
- `train "<goal>" --data <path> --brain <provider:model>` (built-in agent; see docs/AGENT.md)
- `mcp` (MCP server over stdio), `data mark | list | profile`
- `version`, `doctor`, `schema [Name]`
- `plan validate`
- `run start | status | list | pause | resume | stop | retry | logs | metrics | audit`
- `pack search | install | info | list | remove | validate`
- `index entry | add | validate` (for index maintainers)
