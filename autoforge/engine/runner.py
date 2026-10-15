"""Run lifecycle: create -> start (detached worker) -> pause/resume/stop -> finished.

Pause/resume suspend the worker's whole process tree (psutil, works on Linux
and Windows), so they work even with trainers that never check in.
Stop asks nicely first (ctx.check_stop()), then terminates after a grace period.
"""

from __future__ import annotations

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
from autoforge.engine.validate import require_valid
from autoforge.errors import AutoForgeError
from autoforge.schemas.core import HardwareProfile, RunStatus, TrainingPlan


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
    info = store.proc()
    if not info:
        return None
    try:
        p = psutil.Process(info["pid"])
        if abs(p.create_time() - info["create_time"]) > 1 or p.status() == psutil.STATUS_ZOMBIE:
            return None
        return p
    except psutil.Error:
        return None


def _tree(p: psutil.Process) -> list[psutil.Process]:
    try:
        return [p, *p.children(recursive=True)]
    except psutil.Error:
        return [p]


# ---------- create / start ----------


def create_run(
    plan: TrainingPlan,
    actor: str = "human",
    hardware: HardwareProfile | None = None,
    resume_from: str | None = None,
    retry_of: str | None = None,
) -> RunStore:
    check = require_valid(plan, hardware)
    effective = plan.model_copy(update={"hyperparameters": check.hyperparameters})
    store = RunStore(runs_dir() / new_run_id(plan))
    store.dir.mkdir(parents=True)

    entrypoint = check.entrypoint
    if plan.custom_code_path:  # snapshot AI-written code into the run so it is versioned
        file, func = entrypoint.rsplit(":", 1)
        code_dir = store.dir / "code"
        code_dir.mkdir()
        dest = code_dir / Path(file).name
        shutil.copy2(file, dest)
        entrypoint = f"{dest}:{func}"

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
    )
    store.write_record(rec)
    store.write_state(RunState())
    store.write_control(paused=False, stop_requested=False)
    rec.git_tracked = store.git_commit("create run", actor)
    store.write_record(rec)
    store.audit(actor, "create", warnings=check.warnings, retry_of=retry_of)
    return store


def start(store: RunStore, actor: str = "human") -> None:
    if store.proc() is not None:
        raise RunStateError(f"run {store.run_id} was already started")
    log = (store.dir / "logs.txt").open("ab")
    kwargs: dict = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "cwd": store.dir}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True  # survive the CLI exiting
    proc = subprocess.Popen([sys.executable, "-m", "autoforge.engine.worker", str(store.dir)], **kwargs)
    log.close()
    try:
        create_time = psutil.Process(proc.pid).create_time()
    except psutil.Error:
        create_time = 0.0
    store.write_proc(proc.pid, create_time)
    store.audit(actor, "start", pid=proc.pid)


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


def retry(store: RunStore, actor: str = "human") -> RunStore:
    """New run with the same plan; trainers can resume from the old run's checkpoints."""
    rec = store.record()
    ckpt = store.dir / "checkpoints"
    has_ckpt = ckpt.is_dir() and any(ckpt.iterdir())
    plan = rec.plan
    if plan.custom_code_path:
        file, func = rec.entrypoint.rsplit(":", 1)  # use the snapshot that actually ran
        plan = plan.model_copy(update={"custom_code_path": f"{file}:{func}"})
    return create_run(plan, actor, resume_from=str(ckpt) if has_ckpt else None, retry_of=rec.run_id)


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
