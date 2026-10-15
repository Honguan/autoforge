"""On-disk layout of a run. Each file has exactly one writer, so the CLI and
the worker process never race on the same file.

runs/<run_id>/
  run.json       written once at creation (plan, resolved pack/device, actor)
  proc.json      launcher: worker pid + create_time
  state.json     worker: status, timestamps, result, error
                 (the controller writes it only after the worker is dead)
  control.json   controller: {"paused": bool, "stop_requested": bool}
  events.jsonl   worker: metric and log events
  audit.jsonl    every action with its actor
  logs.txt       worker stdout/stderr
  plan.yaml      the plan (git-tracked)
  code/          AI-written trainer code, if any (git-tracked)
  output/        model artifacts
  checkpoints/
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from autoforge import config
from autoforge.errors import AutoForgeError
from autoforge.schemas.core import RunStatus, TrainingPlan

TERMINAL = {RunStatus.succeeded, RunStatus.failed, RunStatus.stopped}
GITIGNORE = "output/\ncheckpoints/\nlogs.txt\nevents.jsonl\naudit.jsonl\nstate.json\nproc.json\ncontrol.json\n"


class RunNotFound(AutoForgeError):
    code = "RUN_NOT_FOUND"
    hint = "Use `autoforge run list` to see runs."


class RunRecord(BaseModel):
    """run.json: immutable facts about a run."""

    run_id: str
    created_at: str
    actor: str
    plan: TrainingPlan
    device: str
    entrypoint: str = Field(description="absolute 'file.py:function'")
    pack_version: str | None = None
    pack_source: str | None = None
    resume_from: str | None = None
    retry_of: str | None = None
    git_tracked: bool = False


class RunState(BaseModel):
    """state.json"""

    status: RunStatus = RunStatus.pending
    started_at: str | None = None
    finished_at: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    model_path: str | None = None
    error: str | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def runs_dir() -> Path:
    return config.home() / "runs"


def new_run_id(plan: TrainingPlan) -> str:
    label = plan.recipe or "custom"
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{label.replace('_', '-')}-{secrets.token_hex(2)}"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _append(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # a partially written last line
    return rows


class RunStore:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)

    @classmethod
    def open(cls, run_id: str) -> "RunStore":
        d = runs_dir() / run_id
        if not (d / "run.json").is_file():
            raise RunNotFound(f"no run with id '{run_id}'")
        return cls(d)

    @property
    def run_id(self) -> str:
        return self.dir.name

    # ---- run.json ----
    def write_record(self, rec: RunRecord) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.dir / "run.json", rec.model_dump_json(indent=2))

    def record(self) -> RunRecord:
        return RunRecord.model_validate_json((self.dir / "run.json").read_text(encoding="utf-8"))

    # ---- state.json ----
    def write_state(self, state: RunState) -> None:
        _atomic_write(self.dir / "state.json", state.model_dump_json(indent=2))

    def state(self) -> RunState:
        p = self.dir / "state.json"
        return RunState.model_validate_json(p.read_text(encoding="utf-8")) if p.is_file() else RunState()

    # ---- proc.json ----
    def write_proc(self, pid: int, create_time: float) -> None:
        _atomic_write(self.dir / "proc.json", json.dumps({"pid": pid, "create_time": create_time}))

    def proc(self) -> dict | None:
        p = self.dir / "proc.json"
        return json.loads(p.read_text()) if p.is_file() else None

    # ---- control.json ----
    def control(self) -> dict:
        p = self.dir / "control.json"
        return json.loads(p.read_text()) if p.is_file() else {"paused": False, "stop_requested": False}

    def write_control(self, **changes: bool) -> None:
        _atomic_write(self.dir / "control.json", json.dumps(self.control() | changes))

    # ---- events / audit ----
    def events(self) -> list[dict]:
        return _read_jsonl(self.dir / "events.jsonl")

    def audit(self, actor: str, action: str, **details: Any) -> None:
        _append(self.dir / "audit.jsonl", {"t": now_iso(), "actor": actor, "action": action, "details": details})

    def audit_log(self) -> list[dict]:
        return _read_jsonl(self.dir / "audit.jsonl")

    def logs_tail(self, n: int) -> list[str]:
        p = self.dir / "logs.txt"
        return p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:] if p.is_file() else []

    # ---- git versioning of plan + code ----
    def git_commit(self, message: str, actor: str) -> bool:
        def git(*args: str) -> None:
            subprocess.run(["git", *args], cwd=self.dir, check=True, capture_output=True, timeout=30)

        try:
            if not (self.dir / ".git").is_dir():
                git("init", "-q")
                (self.dir / ".gitignore").write_text(GITIGNORE)
            git("add", "-A")
            git("-c", "user.name=autoforge", "-c", "user.email=autoforge@localhost",
                "commit", "-q", "--allow-empty", "-m", f"[{actor}] {message}")
            return True
        except (OSError, subprocess.SubprocessError):
            return False


def list_runs() -> list[RunStore]:
    d = runs_dir()
    if not d.is_dir():
        return []
    stores = [RunStore(p) for p in d.iterdir() if (p / "run.json").is_file()]
    return sorted(stores, key=lambda s: s.run_id, reverse=True)


def wait_for(predicate, timeout: float, interval: float = 0.1) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
