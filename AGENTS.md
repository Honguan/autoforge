# AGENTS.md — 給 AI 的使用說明

You are operating AutoForge, a tool that trains AI models on the user's machine.

## Rules
1. Always call `autoforge doctor --json` first to learn the hardware.
2. Every command accepts `--json` and returns `{"ok": bool, "data": ..., "error": {"code","message","hint"}}`.
   Branch on `error.code`, and follow `error.hint`.
3. Read data contracts with `autoforge schema <Name> --json` instead of guessing fields.
4. Privacy: if an error has code `PRIVACY_DENIED`, do not retry. Tell the user the dataset
   is sensitive and needs a local-model brain.
5. Long tasks (training) are asynchronous: start, then poll status. (Available from P3.)

## Finding and learning a training method
1. `autoforge pack search [keywords] --task <task-type> --fit --json`
   `--fit` keeps only recipes this machine can run. Each hit lists `runnable_recipes`.
2. `autoforge pack install <name> --json` installs the commit pinned in the index.
3. `autoforge pack info <name> --json` returns the manifest, recipes (defaults, search space)
   and `knowledge` (KNOWLEDGE.md). **Read `knowledge` before planning.**
4. If `missing_dependencies` is non-empty, tell the user to `pip install` them.

## Error codes
| code | meaning | what to do |
|---|---|---|
| PACK_NOT_FOUND | name not in index / not installed | search again |
| PACK_EXISTS | already installed | use it, or reinstall with `--force` |
| PACK_INVALID | pack fails validation; `data.problems` lists why | do not use it |
| SHA_MISMATCH | source changed after indexing | do not install; tell the user |
| INDEX_UNAVAILABLE | index unreachable and no cache | retry later or set AUTOFORGE_INDEX_URL |
| PACK_SOURCE_ERROR | bad source string or git failure | check the source |

## Commands available now (P2)
- `version`, `doctor`, `schema [Name]`
- `pack search | install | info | list | remove | validate`
- `index entry | add | validate` (for index maintainers)
