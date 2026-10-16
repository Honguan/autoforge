"""Hyperparameter search (P5) with Optuna's ask/tell API.

Every trial is an ordinary Run (same validation, privacy guard, audit, logs and
controls). Trial 0 is the plan's own hyperparameters, so the result always
shows whether tuning beat the starting point.

studies/<study_id>/
  study.json    config (written once)
  state.json    worker: status, best, baseline, current run
  trials.jsonl  worker: one row per finished trial
  control.json  controller: {"stop_requested": bool}
  proc.json     launcher: worker pid
  optuna.db     Optuna storage (sqlite)
  logs.txt
Run as a worker: python -m autoforge.engine.hpo <study_dir>
"""

from __future__ import annotations

import json
import math
import re
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from autoforge import config
from autoforge.engine import runner
from autoforge.engine.store import TERMINAL, RunStore, _append, _atomic_write, _read_jsonl, now_iso
from autoforge.engine.validate import PlanInvalid, require_valid
from autoforge.errors import AutoForgeError
from autoforge.packs import installer
from autoforge.policy.privacy import brain_from_actor
from autoforge.schemas.core import BrainKind, RunStatus, TrainingPlan

MINIMIZE_HINTS = ("loss", "rmse", "mse", "mae", "error", "perplexity")
MINIMIZE_WORDS = {"wer", "cer", "mape", "wape", "smape"}  # whole words only: "wer" must not match "lower"


class StudyNotFound(AutoForgeError):
    code = "STUDY_NOT_FOUND"
    hint = "Use `autoforge hpo list` to see studies."


class HPOUnavailable(AutoForgeError):
    code = "HPO_UNAVAILABLE"
    hint = "pip install optuna"


class StudyConfig(BaseModel):
    study_id: str
    created_at: str
    actor: str
    brain: BrainKind
    plan: TrainingPlan
    metric: str
    direction: Literal["maximize", "minimize"]
    n_trials: int
    timeout_minutes: float | None = None
    search_space: dict[str, dict[str, Any]]
    baseline_params: dict[str, Any] = Field(default_factory=dict)


class StudyState(BaseModel):
    status: Literal["pending", "running", "succeeded", "stopped", "failed"] = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    completed: int = 0
    failed: int = 0
    current_run_id: str | None = None
    best_value: float | None = None
    best_params: dict[str, Any] | None = None
    best_run_id: str | None = None
    baseline_value: float | None = None
    error: str | None = None


class StudyView(StudyState):
    study_id: str
    metric: str
    direction: str
    n_trials: int
    improvement: float | None = Field(default=None, description="best - baseline (sign follows direction)")
    trials: list[dict] = Field(default_factory=list)


def studies_dir() -> Path:
    return config.home() / "studies"


def infer_direction(metric: str) -> str:
    name = metric.lower()
    words = set(re.split(r"[^a-z0-9]+", name))
    return "minimize" if any(h in name for h in MINIMIZE_HINTS) or words & MINIMIZE_WORDS else "maximize"


def _validate_space(space: dict[str, Any]) -> list[str]:
    problems = []
    for name, spec in space.items():
        if not isinstance(spec, dict):
            problems.append(f"{name}: must be a mapping")
        elif "choices" in spec:
            if not isinstance(spec["choices"], list) or not spec["choices"]:
                problems.append(f"{name}: choices must be a non-empty list")
        elif spec.get("type") in ("int", "float"):
            if not ("low" in spec and "high" in spec) or spec["low"] > spec["high"]:
                problems.append(f"{name}: needs low <= high")
            elif spec.get("log") and spec["low"] <= 0:
                problems.append(f"{name}: log scale needs low > 0")
        else:
            problems.append(f"{name}: use {{type: int|float, low, high[, log]}} or {{choices: [...]}}")
    return problems


def _in_space(value: Any, spec: dict[str, Any]) -> bool:
    if "choices" in spec:
        return value in spec["choices"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if spec["type"] == "int" and not float(value).is_integer():
        return False
    return spec["low"] <= value <= spec["high"]


class StudyStore:
    def __init__(self, d: Path):
        self.dir = Path(d)

    @classmethod
    def open(cls, study_id: str) -> "StudyStore":
        d = studies_dir() / study_id
        if not (d / "study.json").is_file():
            raise StudyNotFound(f"no study with id '{study_id}'")
        return cls(d)

    @property
    def study_id(self) -> str:
        return self.dir.name

    def config(self) -> StudyConfig:
        return StudyConfig.model_validate_json((self.dir / "study.json").read_text(encoding="utf-8"))

    def state(self) -> StudyState:
        p = self.dir / "state.json"
        return StudyState.model_validate_json(p.read_text(encoding="utf-8")) if p.is_file() else StudyState()

    def write_state(self, s: StudyState) -> None:
        _atomic_write(self.dir / "state.json", s.model_dump_json(indent=2))

    def trials(self) -> list[dict]:
        return _read_jsonl(self.dir / "trials.jsonl")

    def add_trial(self, row: dict) -> None:
        _append(self.dir / "trials.jsonl", row)

    def control(self) -> dict:
        p = self.dir / "control.json"
        return json.loads(p.read_text()) if p.is_file() else {}

    def proc(self) -> dict | None:
        p = self.dir / "proc.json"
        return json.loads(p.read_text()) if p.is_file() else None


# ---------- create / start ----------


def create_study(
    plan: TrainingPlan,
    metric: str,
    n_trials: int = 20,
    actor: str = "human",
    brain: BrainKind | None = None,
    direction: str | None = None,
    search_space: dict[str, Any] | None = None,
    timeout_minutes: float | None = None,
) -> StudyStore:
    try:
        import optuna  # noqa: F401
    except ImportError:
        raise HPOUnavailable("optuna is not installed")
    brain = brain or brain_from_actor(actor)
    runner.require_data_access(plan, actor, brain)
    check = require_valid(plan)
    if search_space is None:
        if not plan.pack:
            raise PlanInvalid("custom-code plans need an explicit search_space",
                              details={"problems": ["pass search_space for custom_code_path plans"]})
        _, pack = installer.get_installed(plan.pack)
        search_space = pack.recipe(plan.recipe).search_space
    problems = _validate_space(search_space)
    if not search_space:
        problems.append("search space is empty: the recipe declares none; pass search_space")
    if n_trials < 2:
        problems.append("n_trials must be at least 2 (trial 0 is the baseline)")
    if problems:
        raise PlanInvalid("invalid hyperparameter search", details={"problems": problems})

    baseline = {k: v for k, v in check.hyperparameters.items() if k in search_space and _in_space(v, search_space[k])}
    sid = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-hpo-{secrets.token_hex(2)}"
    store = StudyStore(studies_dir() / sid)
    store.dir.mkdir(parents=True)
    cfg = StudyConfig(
        study_id=sid, created_at=now_iso(), actor=actor, brain=brain, plan=plan, metric=metric,
        direction=direction or infer_direction(metric), n_trials=n_trials, timeout_minutes=timeout_minutes,
        search_space=search_space, baseline_params=baseline,
    )
    _atomic_write(store.dir / "study.json", cfg.model_dump_json(indent=2))
    store.write_state(StudyState())
    return store


def start_study(store: StudyStore) -> None:
    pid, ct = runner.spawn_detached("autoforge.engine.hpo", store.dir)
    _atomic_write(store.dir / "proc.json", json.dumps({"pid": pid, "create_time": ct}))


# ---------- status / control ----------


def status(store: StudyStore) -> StudyView:
    cfg, st = store.config(), store.state()
    if st.status in ("pending", "running") and store.proc() and runner.process_alive(store.proc()) is None:
        st = store.state()
        if st.status in ("pending", "running"):
            st.status = "stopped" if store.control().get("stop_requested") else "failed"
            st.error = st.error or (None if st.status == "stopped" else "study worker exited unexpectedly; see logs.txt")
            st.finished_at = now_iso()
            store.write_state(st)
    improvement = None
    if st.best_value is not None and st.baseline_value is not None:
        diff = st.best_value - st.baseline_value
        improvement = diff if cfg.direction == "maximize" else -diff
    return StudyView(**st.model_dump(), study_id=cfg.study_id, metric=cfg.metric, direction=cfg.direction,
                     n_trials=cfg.n_trials, improvement=improvement, trials=store.trials())


def stop(store: StudyStore, actor: str = "human") -> StudyView:
    _atomic_write(store.dir / "control.json", json.dumps({"stop_requested": True, "by": actor}))
    st = store.state()
    if st.current_run_id:
        try:
            runner.stop(RunStore.open(st.current_run_id), actor, grace_seconds=10)
        except AutoForgeError:
            pass
    from autoforge.engine.store import wait_for

    wait_for(lambda: runner.process_alive(store.proc()) is None, 30, 0.2)
    return status(store)


def wait(store: StudyStore, timeout: float) -> StudyView:
    from autoforge.engine.store import wait_for

    wait_for(lambda: status(store).status not in ("pending", "running"), timeout, 0.5)
    return status(store)


def list_studies() -> list[StudyStore]:
    d = studies_dir()
    if not d.is_dir():
        return []
    return sorted((StudyStore(p) for p in d.iterdir() if (p / "study.json").is_file()),
                  key=lambda s: s.study_id, reverse=True)


# ---------- worker ----------


def _suggest(trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params = {}
    for name, spec in space.items():
        if "choices" in spec:
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log")))
        else:
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log")))
    return params


def _better(a: float, b: float | None, direction: str) -> bool:
    return b is None or (a > b if direction == "maximize" else a < b)


def run_study(study_dir: Path) -> int:
    import optuna
    from optuna.trial import TrialState

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    store = StudyStore(study_dir)
    cfg = store.config()
    st = store.state()
    st.status, st.started_at = "running", now_iso()
    store.write_state(st)
    study = optuna.create_study(
        storage=f"sqlite:///{store.dir / 'optuna.db'}", study_name=cfg.study_id,
        direction=cfg.direction, load_if_exists=True,
    )
    if cfg.baseline_params:
        study.enqueue_trial(cfg.baseline_params)
    deadline = time.monotonic() + cfg.timeout_minutes * 60 if cfg.timeout_minutes else None

    try:
        for n in range(cfg.n_trials):
            if store.control().get("stop_requested") or (deadline and time.monotonic() > deadline):
                break
            trial = study.ask()
            params = _suggest(trial, cfg.search_space)
            plan = cfg.plan.model_copy(update={"hyperparameters": cfg.plan.hyperparameters | params})
            row: dict[str, Any] = {"number": trial.number, "params": params, "baseline": n == 0 and bool(cfg.baseline_params)}
            try:
                run = runner.create_run(plan, cfg.actor, brain=cfg.brain, study_id=cfg.study_id)
                runner.start(run, cfg.actor)
                st.current_run_id = run.run_id
                store.write_state(st)
                view = runner.wait(run)
                row |= {"run_id": run.run_id, "status": view.status.value}
                value = view.metrics.get(cfg.metric)
                if view.status is RunStatus.succeeded and value is not None and math.isfinite(value):
                    study.tell(trial, value)
                    row["value"] = value
                    st.completed += 1
                    if row["baseline"]:
                        st.baseline_value = value
                    if _better(value, st.best_value, cfg.direction):
                        st.best_value, st.best_params, st.best_run_id = value, params, run.run_id
                else:
                    study.tell(trial, state=TrialState.FAIL)
                    row["error"] = view.error or f"metric '{cfg.metric}' missing from run metrics {sorted(view.metrics)}"
                    st.failed += 1
            except AutoForgeError as e:
                study.tell(trial, state=TrialState.FAIL)
                row["error"] = f"{e.code}: {e}"
                st.failed += 1
            store.add_trial(row)
            st.current_run_id = None
            store.write_state(st)
            print(f"[hpo] trial {trial.number}: {row.get('value', row.get('error'))}", flush=True)
        if store.control().get("stop_requested"):
            st.status = "stopped"
        elif st.completed == 0:
            st.status, st.error = "failed", st.error or "no trial produced the metric"
        else:
            st.status = "succeeded"
    except BaseException as e:  # noqa: BLE001
        st.status, st.error = "failed", f"{type(e).__name__}: {e}"
    st.finished_at = now_iso()
    store.write_state(st)
    return 0 if st.status == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(run_study(Path(sys.argv[1])))
