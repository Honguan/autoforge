"""Run lifecycle: create -> start (detached worker) -> pause/resume/stop -> finished.

Pause/resume suspend the worker's whole process tree (psutil, works on Linux
and Windows), so they work even with trainers that never check in.
Stop asks nicely first (ctx.check_stop()), then terminates after a grace period.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
import yaml
from pydantic import BaseModel, Field

from autoforge.engine.store import (
    TERMINAL,
    RunRecord,
    RunState,
    RunStore,
    list_runs,
    new_run_id,
    now_iso,
    runs_dir,
    wait_for,
)
from autoforge.data.registry import sensitivity_of
from autoforge.engine.validate import require_valid
from autoforge.errors import AutoForgeError
from autoforge.policy.privacy import brain_from_actor, require_access
from autoforge.schemas.core import BrainKind, HardwareProfile, RunStatus, TrainingPlan


class RunStateError(AutoForgeError):
    code = "RUN_STATE_ERROR"


class RunView(BaseModel):
    """What `run status` returns: a compact, AI-friendly summary."""

    run_id: str
    status: RunStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    actor: str
    pack: str | None = None
    recipe: str | None = None
    device: str
    metrics: dict[str, float] = Field(default_factory=dict)
    latest_metrics: dict[str, float] = Field(default_factory=dict, description="last logged value per metric")
    model_path: str | None = None
    error: str | None = None
    run_dir: str


# ---------- process helpers ----------


def _process(store: RunStore) -> psutil.Process | None:
    return process_alive(store.proc())


def _tree(p: psutil.Process) -> list[psutil.Process]:
    try:
        return [p, *p.children(recursive=True)]
    except psutil.Error:
        return [p]


# ---------- create / start ----------


def _defines(path: Path, *funcs: str) -> bool:
    import ast

    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    return all(f in names for f in funcs)


def require_data_access(plan: TrainingPlan, actor: str, brain: BrainKind | None = None) -> None:
    """Privacy guard (requirement #8): sensitive data only under a local model or a human."""
    b = brain or brain_from_actor(actor)
    require_access(b, sensitivity_of(plan.data_path), plan.data_path)


def create_run(
    plan: TrainingPlan,
    actor: str = "human",
    hardware: HardwareProfile | None = None,
    resume_from: str | None = None,
    retry_of: str | None = None,
    brain: BrainKind | None = None,
    study_id: str | None = None,
) -> RunStore:
    require_data_access(plan, actor, brain)
    check = require_valid(plan, hardware)
    effective = plan.model_copy(update={"hyperparameters": check.hyperparameters})
    store = RunStore(runs_dir() / new_run_id(plan))
    store.dir.mkdir(parents=True)

    entrypoint, predictor, exporters = check.entrypoint, None, {}
    if plan.custom_code_path:  # snapshot AI-written code into the run so it is versioned
        file, func = entrypoint.rsplit(":", 1)
        code_dir = store.dir / "code"
        code_dir.mkdir()
        dest = code_dir / Path(file).name
        shutil.copy2(file, dest)
        entrypoint = f"{dest}:{func}"
        if _defines(dest, "load_model", "predict"):
            predictor = str(dest)
    elif check.pack_root:
        # Snapshot the pack's code: training, serving and export all use exactly this version.
        snap = store.dir / "pack"
        shutil.copytree(check.pack_root, snap, ignore=shutil.ignore_patterns("tests", ".git", "__pycache__"))
        root = Path(check.pack_root).resolve()
        file, func = entrypoint.rsplit(":", 1)
        entrypoint = f"{snap / Path(file).resolve().relative_to(root)}:{func}"
        if check.predictor:
            predictor = str(snap / check.predictor)
        exporters = {fmt: str(snap / ep.rsplit(":", 1)[0]) + ":" + ep.rsplit(":", 1)[1]
                     for fmt, ep in check.exporters.items()}

    (store.dir / "plan.yaml").write_text(
        yaml.safe_dump(effective.model_dump(mode="json"), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    rec = RunRecord(
        run_id=store.run_id,
        created_at=now_iso(),
        actor=actor,
        plan=effective,
        device=check.device,
        entrypoint=entrypoint,
        pack_version=check.pack_version,
        pack_source=check.pack_source,
        resume_from=resume_from,
        retry_of=retry_of,
        study_id=study_id,
        predictor=predictor,
        exporters=exporters,
    )
    store.write_record(rec)
    store.write_state(RunState())
    store.write_control(paused=False, stop_requested=False)
    rec.git_tracked = store.git_commit("create run", actor)
    store.write_record(rec)
    store.audit(actor, "create", warnings=check.warnings, retry_of=retry_of, study_id=study_id)
    return store


def spawn_detached(module: str, workdir: Path) -> tuple[int, float]:
    """Start `python -m <module> <workdir>` detached from the caller; output goes to workdir/logs.txt."""
    log = (workdir / "logs.txt").open("ab")
    kwargs: dict = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "cwd": workdir}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True  # survive the CLI exiting
    proc = subprocess.Popen([sys.executable, "-m", module, str(workdir)], **kwargs)
    log.close()
    try:
        create_time = psutil.Process(proc.pid).create_time()
    except psutil.Error:
        create_time = 0.0
    return proc.pid, create_time


def process_alive(info: dict | None) -> psutil.Process | None:
    """The process described by {"pid", "create_time"} if it is still running (not a zombie)."""
    if not info:
        return None
    try:
        p = psutil.Process(info["pid"])
        if abs(p.create_time() - info["create_time"]) > 1 or p.status() == psutil.STATUS_ZOMBIE:
            return None
        return p
    except psutil.Error:
        return None


def start(store: RunStore, actor: str = "human") -> None:
    if store.proc() is not None:
        raise RunStateError(f"run {store.run_id} was already started")
    pid, create_time = spawn_detached("autoforge.engine.worker", store.dir)
    store.write_proc(pid, create_time)
    store.audit(actor, "start", pid=pid)


# ---------- status ----------


def status(store: RunStore) -> RunView:
    rec, state = store.record(), store.state()
    if state.status not in TERMINAL and store.proc() is not None and _process(store) is None:
        # Worker is gone but never wrote a final state: it crashed or was killed.
        state = store.state()  # re-read: it may have finished just now
        if state.status not in TERMINAL:
            stopped = store.control().get("stop_requested")
            state.status = RunStatus.stopped if stopped else RunStatus.failed
            state.error = state.error or (None if stopped else "worker process exited unexpectedly; see logs")
            state.finished_at = now_iso()
            store.write_state(state)
    shown = state.status
    if shown is RunStatus.running and store.control().get("paused"):
        shown = RunStatus.paused
    latest: dict[str, float] = {}
    for e in store.events():
        if e.get("type") == "metric":
            latest[e["name"]] = e["value"]
    return RunView(
        run_id=store.run_id,
        status=shown,
        created_at=rec.created_at,
        started_at=state.started_at,
        finished_at=state.finished_at,
        actor=rec.actor,
        pack=rec.plan.pack,
        recipe=rec.plan.recipe,
        device=rec.device,
        metrics=state.metrics,
        latest_metrics=latest,
        model_path=state.model_path,
        error=state.error,
        run_dir=str(store.dir),
    )


def all_runs() -> list[RunView]:
    return [status(s) for s in list_runs()]


def wait(store: RunStore, timeout: float | None = None) -> RunView:
    wait_for(lambda: status(store).status in TERMINAL, timeout if timeout is not None else 10**9, 0.2)
    return status(store)


# ---------- control ----------


def _require_live(store: RunStore, action: str) -> psutil.Process:
    view = status(store)
    if view.status in TERMINAL:
        raise RunStateError(f"cannot {action}: run {store.run_id} already {view.status.value}")
    p = _process(store)
    if p is None:
        raise RunStateError(f"cannot {action}: run {store.run_id} has no live worker (status {view.status.value})")
    return p


def pause(store: RunStore, actor: str = "human") -> RunView:
    p = _require_live(store, "pause")
    for proc in _tree(p):
        try:
            proc.suspend()
        except psutil.Error:
            pass
    store.write_control(paused=True)
    store.audit(actor, "pause")
    return status(store)


def resume(store: RunStore, actor: str = "human") -> RunView:
    p = _require_live(store, "resume")
    for proc in _tree(p):
        try:
            proc.resume()
        except psutil.Error:
            pass
    store.write_control(paused=False)
    store.audit(actor, "resume")
    return status(store)


def stop(store: RunStore, actor: str = "human", grace_seconds: float = 10.0) -> RunView:
    p = _require_live(store, "stop")
    store.write_control(stop_requested=True)
    store.audit(actor, "stop", grace_seconds=grace_seconds)
    if store.control().get("paused"):
        for proc in _tree(p):
            try:
                proc.resume()
            except psutil.Error:
                pass
        store.write_control(paused=False)
    if not wait_for(lambda: _process(store) is None, grace_seconds):
        procs = _tree(p)
        for proc in procs:
            try:
                proc.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(procs, timeout=5)
        for proc in alive:
            try:
                proc.kill()
            except psutil.Error:
                pass
        store.audit("system", "terminate", reason="grace period expired")
    wait_for(lambda: _process(store) is None, 5)
    return status(store)


def adjust(store: RunStore, actor: str, changes: dict) -> dict:
    """Send hyperparameter changes to a live run. Takes effect when the trainer calls
    ctx.check_updates(); every change is audited and committed to the run's git repo."""
    if not changes:
        raise RunStateError("no changes given")
    _require_live(store, "adjust")
    control = store.control()
    overrides = (control.get("overrides") or {}) | changes
    seq = int(control.get("adjust_seq", 0)) + 1
    store.write_control(overrides=overrides, adjust_seq=seq)
    (store.dir / "adjustments.yaml").write_text(
        yaml.safe_dump(overrides, allow_unicode=True, sort_keys=True), encoding="utf-8")
    store.git_commit(f"adjust {changes}", actor)
    store.audit(actor, "adjust", changes=changes, seq=seq)
    return {"run_id": store.run_id, "overrides": overrides, "seq": seq,
            "note": "applied when the trainer next calls ctx.check_updates(); trainers that never call it ignore changes"}


def compare(run_ids: list[str], metric: str | None = None) -> dict:
    """Side-by-side metrics and hyperparameters; sorted by `metric` (best first) when given."""
    from autoforge.engine.hpo import infer_direction

    rows = []
    for rid in run_ids:
        s = RunStore.open(rid)
        v, rec = status(s), s.record()
        rows.append({"run_id": rid, "status": v.status.value, "recipe": v.recipe or "custom",
                     "hyperparameters": rec.plan.hyperparameters, "metrics": v.metrics or v.latest_metrics})
    keys = sorted({k for r in rows for k in r["hyperparameters"]})
    differing = [k for k in keys if len({json.dumps(r["hyperparameters"].get(k), sort_keys=True) for r in rows}) > 1]
    best = None
    if metric:
        scored = [r for r in rows if metric in r["metrics"]]
        rev = infer_direction(metric) == "maximize"
        scored.sort(key=lambda r: r["metrics"][metric], reverse=rev)
        rows = scored + [r for r in rows if metric not in r["metrics"]]
        best = scored[0]["run_id"] if scored else None
    return {"metric": metric, "best_run_id": best, "differing_hyperparameters": differing, "runs": rows}


def retry(store: RunStore, actor: str = "human", brain: BrainKind | None = None) -> RunStore:
    """New run with the same plan; trainers can resume from the old run's checkpoints."""
    rec = store.record()
    ckpt = store.dir / "checkpoints"
    has_ckpt = ckpt.is_dir() and any(ckpt.iterdir())
    plan = rec.plan
    if plan.custom_code_path:
        file, func = rec.entrypoint.rsplit(":", 1)  # use the snapshot that actually ran
        plan = plan.model_copy(update={"custom_code_path": f"{file}:{func}"})
    return create_run(plan, actor, resume_from=str(ckpt) if has_ckpt else None, retry_of=rec.run_id, brain=brain)


def metrics_summary(store: RunStore, last: int = 5) -> dict:
    """Per metric: last value, best (max and min), count and the last N values."""
    series: dict[str, list[dict]] = {}
    for e in store.events():
        if e.get("type") == "metric":
            series.setdefault(e["name"], []).append(e)
    out = {}
    for name, evs in series.items():
        values = [e["value"] for e in evs]
        out[name] = {
            "last": values[-1],
            "max": max(values),
            "min": min(values),
            "count": len(values),
            "recent": [{"step": e.get("step"), "value": e["value"]} for e in evs[-last:]],
        }
    return out
