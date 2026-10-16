"""TrainContext: the object every trainer receives.

Trainer contract (v1):
    def train(ctx: TrainContext) -> dict:
        ...
        return {"metrics": {"accuracy": 0.9}, "model_path": "..."}

Trainers should call `ctx.log_metric(...)` as they go and `ctx.check_stop()`
between expensive steps so a stop request can end the run cleanly.
Pause works without trainer cooperation (the process is suspended).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunStopped(Exception):
    """Raised by ctx.check_stop() when a stop was requested."""


class TrainContext:
    def __init__(
        self,
        run_dir: Path,
        data_path: str,
        target: str | None,
        task_type: str,
        hyperparameters: dict[str, Any],
        device: str,
        seed: int = 0,
        resume_from: str | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.data_path = data_path
        self.target = target
        self.task_type = task_type
        self.hyperparameters = hyperparameters
        self.device = device
        self.seed = seed
        self.output_dir = self.run_dir / "output"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.resume_from = Path(resume_from) if resume_from else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._events = self.run_dir / "events.jsonl"
        self._control = self.run_dir / "control.json"
        self._last_check = 0.0
        self._stop = False
        self._adjust_seq = 0
        self._control_mtime: tuple | None = None

    # ---- metrics / logging ----

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        event = {"t": time.time(), "type": "metric", "name": name, "value": float(value), "step": step}
        with self._events.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def log(self, message: str) -> None:
        event = {"t": time.time(), "type": "log", "message": message}
        with self._events.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(message, flush=True)

    # ---- control ----

    def should_stop(self) -> bool:
        """True once a stop was requested. Reads the control file at most every 0.5 s."""
        now = time.monotonic()
        if not self._stop and now - self._last_check >= 0.5:
            self._last_check = now
            try:
                self._stop = bool(json.loads(self._control.read_text()).get("stop_requested"))
            except (OSError, ValueError):
                pass
        return self._stop

    def check_updates(self) -> dict[str, Any]:
        """Hyperparameter changes sent with `run adjust` since the last call (already applied to
        self.hyperparameters). Trainers that support live changes call this between steps; control.json
        is only re-read when it changed, so calling this every step is cheap."""
        try:
            st = self._control.stat()
            # (mtime, size) plus a re-read every 2 s covers filesystems with coarse timestamps
            key = (st.st_mtime_ns, st.st_size, int(time.monotonic() // 2))
            if key == self._control_mtime:
                return {}
            self._control_mtime = key
            control = json.loads(self._control.read_text())
        except (OSError, ValueError):
            return {}
        seq = int(control.get("adjust_seq", 0))
        if seq <= self._adjust_seq:
            return {}
        self._adjust_seq = seq
        changes = dict(control.get("overrides") or {})
        changed = {k: v for k, v in changes.items() if self.hyperparameters.get(k) != v}
        self.hyperparameters.update(changed)
        if changed:
            self.log(f"hyperparameters adjusted: {changed}")
        return changed

    def check_stop(self) -> None:
        if self.should_stop():
            raise RunStopped()

    def latest_checkpoint(self) -> Path | None:
        """Newest file in resume_from (if retrying) or this run's checkpoint dir."""
        for d in (self.resume_from, self.checkpoint_dir):
            if d and d.is_dir():
                files = sorted((p for p in d.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime)
                if files:
                    return files[-1]
        return None

    # ---- compatibility with the provisional dict contract ----

    def __getitem__(self, key: str) -> Any:
        if key in ("data_path", "target", "task_type", "hyperparameters", "seed", "device"):
            return getattr(self, key)
        if key == "output_dir":
            return str(self.output_dir)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default
