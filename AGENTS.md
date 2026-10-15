# AGENTS.md — 給 AI 的使用說明

You are operating AutoForge, a tool that trains AI models on the user's machine.

## Rules
1. Always call `autoforge doctor --json` first to learn the hardware.
2. Every command accepts `--json` and returns `{"ok": bool, "data": ..., "error": {"code","message","hint"}}`.
3. Read data contracts with `autoforge schema <Name> --json` instead of guessing fields.
4. Privacy: if an error has code `PRIVACY_DENIED`, do not retry. Tell the user the dataset
   is sensitive and needs a local-model brain.
5. Long tasks (training) are asynchronous: start, then poll status. (Available from P3.)

## Available now (P1)
- `autoforge version --json`
- `autoforge doctor --json`
- `autoforge schema [Name] --json`
