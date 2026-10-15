"""Worker process: `python -m autoforge.engine.worker <run_dir>`.

Loads the trainer entrypoint, runs it with a TrainContext and records the
outcome in state.json. Runs in its own process so a crash or kill never
takes down the caller.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

from autoforge.engine.context import RunStopped, TrainContext
from autoforge.engine.store import RunState, RunStore, now_iso
from autoforge.schemas.core import RunStatus


def load_entrypoint(entrypoint: str):
    file, func = entrypoint.rsplit(":", 1)
    path = Path(file)
    spec = importlib.util.spec_from_file_location(f"autoforge_trainer_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))  # allow trainers to import sibling modules
    spec.loader.exec_module(module)
    return getattr(module, func)


def run(run_dir: Path) -> int:
    store = RunStore(run_dir)
    rec = store.record()
    state = RunState(status=RunStatus.running, started_at=now_iso())
    store.write_state(state)
    ctx = TrainContext(
        run_dir=store.dir,
        data_path=rec.plan.data_path,
        target=rec.plan.task.target,
        task_type=rec.plan.task.task_type,
        hyperparameters=rec.plan.hyperparameters,
        device=rec.device,
        seed=rec.plan.seed,
        resume_from=rec.resume_from,
    )
    try:
        fn = load_entrypoint(rec.entrypoint)
        result = fn(ctx) or {}
        if not isinstance(result, dict):
            raise TypeError(f"trainer must return a dict, got {type(result).__name__}")
        state.metrics = {k: float(v) for k, v in (result.get("metrics") or {}).items()}
        state.model_path = result.get("model_path")
        for name, value in state.metrics.items():
            ctx.log_metric(name, value, step=None)
        state.status = RunStatus.succeeded
    except RunStopped:
        state.status = RunStatus.stopped
    except BaseException as e:  # noqa: BLE001 - record every failure, including KeyboardInterrupt
        traceback.print_exc()
        state.status = RunStatus.failed
        state.error = f"{type(e).__name__}: {e}"
    state.finished_at = now_iso()
    store.write_state(state)
    print(f"[autoforge] run {store.run_id} finished: {state.status.value}", flush=True)
    return 0 if state.status is RunStatus.succeeded else 1


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1])))
