"""Core API shared by the MCP server (and later REST / GUI).

Functions return JSON-serializable data and raise AutoForgeError subclasses;
transports wrap them in the Envelope shape. `actor` says who acts (for the
audit log); `brain` decides what data the caller may see (privacy guard).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from autoforge import config
from autoforge.data import registry
from autoforge.data.profile import profile
from autoforge.engine import runner
from autoforge.engine.store import RunStore, list_runs
from autoforge.engine.validate import check_plan, parse_plan
from autoforge.errors import AutoForgeError
from autoforge.hardware.detect import detect_hardware
from autoforge.packs import installer
from autoforge.packs.index import load_index, search
from autoforge.policy.privacy import require_access
from autoforge.schemas.core import BrainKind

TRAINER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}\.py$")


class BadRequest(AutoForgeError):
    code = "BAD_REQUEST"


# ---------- hardware / schema ----------


def hardware() -> dict:
    hw = detect_hardware()
    return hw.model_dump() | {"best_device": hw.best_device}


# ---------- packs ----------


def pack_search(query: str | None = None, task: str | None = None, fit: bool = True) -> dict:
    hw = detect_hardware() if fit else None
    index, notes = load_index()
    lock = installer.read_lock()
    hits = [
        h.model_dump(mode="json") | {"installed": h.entry.name in lock}
        for h in search(index, task=task, query=query, hardware=hw)
    ]
    return {"hits": hits, "notes": notes}


def pack_install(source: str, force: bool = False) -> dict:
    return installer.install(source, force=force).model_dump()


def pack_info(name: str, knowledge: bool = True) -> dict:
    rec, pack = installer.get_installed(name)
    data = {
        "installed": rec.model_dump(),
        "manifest": pack.manifest.model_dump(),
        "recipes": [r.model_dump() for r in pack.recipes],
        "missing_dependencies": installer.missing_dependencies(pack),
    }
    if knowledge:
        data["knowledge"] = pack.knowledge()
    return data


def pack_list() -> list[dict]:
    return [r.model_dump() for r in installer.read_lock().values()]


# ---------- data ----------


def data_profile(path: str, brain: BrainKind) -> dict:
    p = Path(path).expanduser()
    if not p.exists():
        raise BadRequest(f"path not found: {path}")
    sens = registry.sensitivity_of(p)
    access = require_access(brain, sens, str(p.resolve()))
    return profile(p, access) | {"sensitivity": sens.value}


# ---------- trainers written by AI ----------


def trainers_dir() -> Path:
    return config.home() / "workspace" / "trainers"


def trainer_write(filename: str, source: str) -> dict:
    """Save AI-written trainer code to the workspace and report its top-level functions."""
    if not TRAINER_NAME_RE.fullmatch(filename):
        raise BadRequest("filename must look like 'my_trainer.py' (letters, digits, '_')")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise BadRequest(f"syntax error: {e}", hint="Fix the code and write it again.")
    path = trainers_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    return {
        "path": str(path),
        "functions": funcs,
        "use_in_plan": f"custom_code_path: {path}:{'train' if 'train' in funcs else (funcs[0] if funcs else 'train')}",
    }


# ---------- plans / runs ----------


def _plan(plan: dict[str, Any], base: Path | None):
    return parse_plan(plan, base or Path.cwd())


def plan_validate(plan: dict[str, Any], actor: str, brain: BrainKind, base: Path | None = None) -> dict:
    p = _plan(plan, base)
    runner.require_data_access(p, actor, brain)
    return check_plan(p).model_dump() | {"resolved_data_path": p.data_path}


def run_start(plan: dict[str, Any], actor: str, brain: BrainKind, base: Path | None = None) -> dict:
    store = runner.create_run(_plan(plan, base), actor, brain=brain)
    runner.start(store, actor)
    return runner.status(store).model_dump(mode="json")


def run_status(run_id: str) -> dict:
    return runner.status(RunStore.open(run_id)).model_dump(mode="json")


def run_wait(run_id: str, timeout_seconds: float = 30) -> dict:
    timeout = max(0.0, min(timeout_seconds, 120.0))
    return runner.wait(RunStore.open(run_id), timeout).model_dump(mode="json")


def run_list(limit: int = 20) -> list[dict]:
    return [runner.status(s).model_dump(mode="json") for s in list_runs()[:limit]]


def run_metrics(run_id: str, last: int = 5) -> dict:
    return runner.metrics_summary(RunStore.open(run_id), last)


def run_logs(run_id: str, tail: int = 50) -> dict:
    return {"lines": RunStore.open(run_id).logs_tail(max(1, min(tail, 500)))}


def run_audit(run_id: str) -> list[dict]:
    return RunStore.open(run_id).audit_log()


def run_pause(run_id: str, actor: str) -> dict:
    return runner.pause(RunStore.open(run_id), actor).model_dump(mode="json")


def run_resume(run_id: str, actor: str) -> dict:
    return runner.resume(RunStore.open(run_id), actor).model_dump(mode="json")


def run_stop(run_id: str, actor: str, grace_seconds: float = 10.0) -> dict:
    return runner.stop(RunStore.open(run_id), actor, grace_seconds).model_dump(mode="json")


def run_retry(run_id: str, actor: str, brain: BrainKind) -> dict:
    store = runner.retry(RunStore.open(run_id), actor, brain=brain)
    runner.start(store, actor)
    return runner.status(store).model_dump(mode="json")


# ---------- P5: diagnosis, live adjustment, comparison, hyperparameter search ----------


def run_diagnose(run_id: str) -> dict:
    from autoforge.engine.diagnose import diagnose

    return diagnose(RunStore.open(run_id))


def run_adjust(run_id: str, actor: str, changes: dict[str, Any]) -> dict:
    return runner.adjust(RunStore.open(run_id), actor, changes)


def run_compare(run_ids: list[str], metric: str | None = None) -> dict:
    if not run_ids:
        raise BadRequest("give at least one run_id")
    return runner.compare(run_ids, metric)


def hpo_start(
    plan: dict[str, Any], metric: str, n_trials: int, actor: str, brain: BrainKind,
    direction: str | None = None, search_space: dict[str, Any] | None = None,
    timeout_minutes: float | None = None, base: Path | None = None,
) -> dict:
    from autoforge.engine import hpo

    if direction not in (None, "maximize", "minimize"):
        raise BadRequest("direction must be 'maximize' or 'minimize'")
    store = hpo.create_study(_plan(plan, base), metric, n_trials, actor, brain, direction, search_space, timeout_minutes)
    hpo.start_study(store)
    return hpo.status(store).model_dump(mode="json")


def hpo_status(study_id: str) -> dict:
    from autoforge.engine import hpo

    return hpo.status(hpo.StudyStore.open(study_id)).model_dump(mode="json")


def hpo_wait(study_id: str, timeout_seconds: float = 60) -> dict:
    from autoforge.engine import hpo

    return hpo.wait(hpo.StudyStore.open(study_id), max(0.0, min(timeout_seconds, 120.0))).model_dump(mode="json")


def hpo_stop(study_id: str, actor: str) -> dict:
    from autoforge.engine import hpo

    return hpo.stop(hpo.StudyStore.open(study_id), actor).model_dump(mode="json")


def hpo_list(limit: int = 20) -> list[dict]:
    from autoforge.engine import hpo

    out = []
    for s in hpo.list_studies()[:limit]:
        v = hpo.status(s).model_dump(mode="json")
        v.pop("trials", None)
        out.append(v)
    return out


# ---------- P6: model registry and serving ----------


def model_register(run_id: str, name: str, actor: str, description: str = "") -> dict:
    from autoforge.registry import models as reg

    return reg.register(run_id, name, actor, description).model_dump(mode="json")


def model_list() -> list[dict]:
    from autoforge.registry import models as reg

    return [{"model": c.ref, "task": c.task.task_type, "target": c.task.target, "metrics": c.metrics,
             "servable": c.servable, "run_id": c.run_id, "versions": reg.versions(c.name)} for c in reg.list_models()]


def model_info(ref: str) -> dict:
    from autoforge.registry import models as reg

    card = reg.get_card(ref)
    return card.model_dump(mode="json") | {"versions": reg.versions(card.name),
                                           "export_formats": ["bundle", *card.exporters]}


def model_predict(ref: str, rows: list[dict[str, Any]]) -> dict:
    from autoforge.registry import models as reg

    return reg.predict(ref, rows)


def model_export(ref: str, fmt: str, out: str | None = None) -> dict:
    from autoforge.registry import models as reg

    card = reg.get_card(ref)
    suffix = {"bundle": "zip"}.get(fmt, fmt)
    path = Path(out) if out else config.home() / "exports" / f"{card.name}-v{card.version}.{suffix}"
    return reg.export(ref, fmt, path)
