"""Pack authoring (P7): scaffold a new pack, let an AI write one from files, run smoke tests.

The intended "generate a pack from a GitHub repo" flow: the AI reads the repo with
its own tools (Claude Code, web access...), writes pack files with `pack_create`,
fixes what `pack validate` reports, runs `pack test`, then installs it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from autoforge import config
from autoforge.errors import AutoForgeError
from autoforge.packs.manifest import NAME_RE, collect_problems

MAX_FILES = 200
MAX_FILE_BYTES = 1_000_000


class AuthoringError(AutoForgeError):
    code = "PACK_AUTHORING_ERROR"


def workspace_packs() -> Path:
    return config.home() / "workspace" / "packs"


def _safe_rel(path: str) -> Path:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts or not p.parts or path.startswith(("/", "\\")):
        raise AuthoringError(f"unsafe path '{path}'", hint="use relative paths inside the pack, e.g. trainers/train.py")
    return p


def create_from_files(name: str, files: dict[str, str], overwrite: bool = False) -> dict:
    """Write a pack (path -> text content) into the workspace and validate it. Nothing is executed."""
    if not NAME_RE.fullmatch(name):
        raise AuthoringError(f"bad pack name '{name}'", hint="lowercase kebab-case, e.g. text-classification")
    if not files or len(files) > MAX_FILES:
        raise AuthoringError(f"give between 1 and {MAX_FILES} files")
    dest = workspace_packs() / name
    if dest.exists():
        if not overwrite:
            raise AuthoringError(f"workspace pack '{name}' exists", hint="pass overwrite=true to replace it")
        shutil.rmtree(dest)
    for rel, content in files.items():
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise AuthoringError(f"{rel} is larger than {MAX_FILE_BYTES} bytes")
        target = dest / _safe_rel(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    pack, problems = collect_problems(dest)
    return {"path": str(dest), "valid": pack is not None, "problems": problems,
            "next": "fix problems and call again with overwrite=true" if problems
            else f"run pack tests, then pack_install('{dest}')"}


def test_pack(path: str | Path, timeout: float = 600) -> dict:
    """Run the pack's smoke tests (tests/) with pytest in a subprocess."""
    root = Path(path).expanduser().resolve()
    tests = root / "tests"
    if not tests.is_dir():
        raise AuthoringError(f"{root} has no tests/ directory", hint="every pack needs a CPU smoke test in tests/")
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(root), str(tests)],
            cwd=root, capture_output=True, text=True, timeout=timeout,
        )
        out, code = proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        out, code = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), -1
    seconds = round(time.monotonic() - start, 1)
    summary = next((line for line in reversed(out.strip().splitlines()) if re.search(r"passed|failed|error|skipped", line)), "")
    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|skipped|error)", summary)}
    # pytest exits 5 when every test was skipped (e.g. torch not installed): skipped, not failed.
    ok = code == 0 or (code == 5 and counts.get("skipped", 0) > 0)
    return {"path": str(root), "ok": ok, "timed_out": code == -1, "seconds": seconds,
            "passed": counts.get("passed", 0), "failed": counts.get("failed", 0) + counts.get("error", 0),
            "skipped": counts.get("skipped", 0), "summary": summary, "output_tail": out.strip().splitlines()[-40:]}


# ---------------------------------------------------------------- scaffold

_TRAINER = '''"""Trainer for the {name} pack. Contract: docs/PACK_SPEC.md §5 (TrainContext)."""

from __future__ import annotations

import json
from pathlib import Path


def train(ctx) -> dict:
    hp = ctx.hyperparameters
    ctx.log(f"training with {{hp}} on {{ctx.device}}")
    # TODO: load ctx.data_path, train, call ctx.log_metric(...) and ctx.check_stop() between steps
    score = 0.0
    out = Path(ctx.output_dir)
    (out / "model.json").write_text(json.dumps({{"hyperparameters": hp}}))
    # Describe the model's inputs for the registry / serving:
    (out / "io.json").write_text(json.dumps({{"features": [{{"name": "x", "dtype": "numeric"}}]}}))
    return {{"metrics": {{"score": score}}, "model_path": str(out / "model.json")}}


def load_model(model_dir):
    return json.loads((Path(model_dir) / "model.json").read_text())


def predict(model, rows: list[dict]) -> list[dict]:
    return [{{"prediction": 0}} for _ in rows]
'''

_RECIPE = """id: baseline
name: Baseline
task_types: [{task}]
entrypoint: trainers/train.py:train
predictor: trainers/train.py
requires: {{ device: cpu, min_ram_gb: 1 }}
good_for: Describe when this recipe is the right choice.
avoid_when: Describe when to pick something else.
defaults: {{ learning_rate: 0.01 }}
search_space:
  learning_rate: {{ type: float, low: 0.0001, high: 0.1, log: true }}
"""

_KNOWLEDGE = """# {name} — knowledge for AI planners

## When to use
- TODO: the data and task this pack is for.

## When NOT to use
- TODO

## Choosing a recipe
| Situation | Recipe |
|---|---|
| TODO | `baseline` |

## Data requirements
- TODO: file format, columns, sizes.

## Metrics
- TODO

## Tuning guide
- TODO: which hyperparameters matter, with concrete ranges.

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| TODO | TODO | TODO |
"""

_TEST = '''"""Smoke test: must pass on CPU in < 30 s."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trainer", ROOT / "trainers" / "train.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def test_train_and_predict(tmp_path):
    from autoforge.engine.context import TrainContext

    data = tmp_path / "data.csv"
    data.write_text("x,y\\n1,0\\n2,1\\n")
    ctx = TrainContext(tmp_path / "run", str(data), "y", "{task}", {{"learning_rate": 0.01}}, "cpu")
    result = trainer.train(ctx)
    assert "score" in result["metrics"]
    model = trainer.load_model(ctx.output_dir)
    assert len(trainer.predict(model, [{{"x": 1}}])) == 1
'''


def scaffold(name: str, task: str, dest_parent: Path) -> dict:
    if not NAME_RE.fullmatch(name):
        raise AuthoringError(f"bad pack name '{name}'")
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", task):
        raise AuthoringError(f"bad task type '{task}'", hint="kebab-case, e.g. text-classification")
    root = Path(dest_parent).expanduser().resolve() / name
    if root.exists():
        raise AuthoringError(f"{root} already exists")
    files = {
        "pack.yaml": f"name: {name}\nversion: 0.1.0\nsummary: TODO one sentence.\ntasks: [{task}]\n"
                     f"recipes: [baseline]\ndependencies: []\nlicense: MIT\n",
        "KNOWLEDGE.md": _KNOWLEDGE.format(name=name),
        "recipes/baseline.yaml": _RECIPE.format(task=task),
        "trainers/train.py": _TRAINER.format(name=name),
        "tests/test_smoke.py": _TEST.format(task=task),
    }
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content, encoding="utf-8")
    pack, problems = collect_problems(root)
    return {"path": str(root), "files": sorted(files), "valid": pack is not None, "problems": problems}
