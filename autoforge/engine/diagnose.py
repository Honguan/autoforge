"""Deterministic run diagnosis (P5). Gives any AI, including small local models,
concrete findings to act on instead of raw logs.

Each finding: {severity: error|warning|info, code, message, suggestion}.
"""

from __future__ import annotations

import math
import re

from autoforge.engine import runner
from autoforge.engine.store import RunStore
from autoforge.packs import installer
from autoforge.packs.errors import PackError
from autoforge.schemas.core import RunStatus

LOG_RULES: list[tuple[str, str, str, str]] = [
    # (regex, code, message, suggestion)
    (r"No module named '([\w.]+)'", "MISSING_PACKAGE", "Python package '{0}' is not installed",
     "pip install {0} (check pack_info missing_dependencies)"),
    (r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED", "GPU_OOM", "GPU ran out of memory",
     "lower batch_size, use a smaller model, enable gradient accumulation or 4-bit/QLoRA"),
    (r"MemoryError|Killed", "RAM_OOM", "the process ran out of RAM",
     "use fewer rows or features, lower batch_size, or drop high-cardinality columns"),
    (r"least populated class in y has only 1 member|The test_size .* should be greater or equal to the number of classes",
     "RARE_CLASS", "a class has too few examples to split", "merge or drop rare classes, or collect more data"),
    (r"could not convert string to float: '([^']*)'", "NON_NUMERIC", "a non-numeric value '{0}' reached a numeric step",
     "encode categorical columns or clean the column containing that value"),
    (r"KeyError: '([^']+)'", "MISSING_COLUMN", "column '{0}' was not found",
     "check task.target and the CSV header (data_profile)"),
    (r"FileNotFoundError: .*?'([^']+)'", "FILE_NOT_FOUND", "file '{0}' was not found", "check data_path"),
]
LOSS_LIKE = ("loss", "rmse", "mse", "mae", "error", "perplexity")


def _is_loss(name: str) -> bool:
    return any(h in name.lower() for h in LOSS_LIKE)


def _finding(severity: str, code: str, message: str, suggestion: str = "") -> dict:
    return {"severity": severity, "code": code, "message": message, "suggestion": suggestion}


def _series(store: RunStore) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for e in store.events():
        if e.get("type") == "metric" and e.get("step") is not None:
            out.setdefault(e["name"], []).append(float(e["value"]))
    return out


def _log_findings(store: RunStore) -> list[dict]:
    text = "\n".join(store.logs_tail(300))
    found = []
    for pattern, code, msg, sug in LOG_RULES:
        m = re.search(pattern, text)
        if m:
            groups = m.groups() or ("",)
            found.append(_finding("error", code, msg.format(*groups), sug.format(*groups)))
    return found


def _curve_findings(series: dict[str, list[float]], tunable: set[str]) -> list[dict]:
    found = []
    lr = next((k for k in ("learning_rate", "lr") if k in tunable), "learning_rate")
    for name, xs in series.items():
        if any(not math.isfinite(x) for x in xs):
            found.append(_finding("error", "DIVERGED", f"'{name}' became NaN/inf",
                                  f"lower {lr} (e.g. ÷10), add gradient clipping, check input scaling"))
            continue
        if len(xs) < 4 or not _is_loss(name):
            continue
        best_i = min(range(len(xs)), key=xs.__getitem__)
        if xs[-1] > xs[best_i] * 1.2 and best_i < len(xs) - 2 and not name.lower().startswith(("val", "valid", "eval", "test")):
            found.append(_finding("warning", "LOSS_RISING", f"'{name}' rose {xs[-1] / xs[best_i]:.2f}× above its best",
                                  f"lower {lr}; the run may be unstable"))
        half = xs[len(xs) // 2:]
        if len(half) >= 3 and half[0] != 0 and abs(half[0] - half[-1]) / abs(half[0]) < 0.01:
            found.append(_finding("info", "PLATEAU", f"'{name}' changed < 1% over the last half of training",
                                  f"stop earlier to save time, or raise {lr} / model capacity if the score is too low"))
    # overfitting: training loss keeps falling while validation loss rises
    for name, val in series.items():
        lname = name.lower()
        if not lname.startswith(("val_", "valid_", "eval_")) or not _is_loss(name) or len(val) < 4:
            continue
        base = name.split("_", 1)[1]
        train = series.get(f"train_{base}") or series.get(base)
        if not train or len(train) < 4:
            continue
        best_i = min(range(len(val)), key=val.__getitem__)
        if best_i < len(val) - 2 and val[-1] > val[best_i] * 1.05 and train[-1] < train[len(train) // 2]:
            reg = [k for k in ("dropout", "weight_decay", "min_samples_leaf", "max_depth", "C") if k in tunable]
            found.append(_finding("warning", "OVERFITTING",
                                  f"'{name}' was best at step {best_i} and then rose while training loss kept falling",
                                  "stop at the best step (early stopping)" + (f"; regularize via {', '.join(reg)}" if reg else "")))
    return found


def diagnose(store: RunStore) -> dict:
    view = runner.status(store)
    rec = store.record()
    tunable: set[str] = set(rec.plan.hyperparameters)
    if rec.plan.pack:
        try:
            _, pack = installer.get_installed(rec.plan.pack)
            recipe = pack.recipe(rec.plan.recipe or "")
            if recipe:
                tunable |= set(recipe.search_space)
        except PackError:
            pass

    findings: list[dict] = []
    if view.status is RunStatus.failed:
        findings += _log_findings(store) or [_finding(
            "error", "UNKNOWN_FAILURE", view.error or "run failed", "read run_logs for the traceback")]
    elif view.status is RunStatus.stopped:
        findings.append(_finding("info", "STOPPED", "the run was stopped before finishing",
                                 "run_retry resumes from checkpoints if the trainer saves them"))
    findings += _curve_findings(_series(store), tunable)

    for name, value in view.metrics.items():
        if name.lower() in ("accuracy", "f1", "f1_macro", "auc", "roc_auc", "r2") and value >= 0.999:
            findings.append(_finding("warning", "POSSIBLE_LEAKAGE", f"{name} = {value:.4f} is suspiciously perfect",
                                     "check whether a feature leaks the target (e.g. an ID or post-outcome column)"))

    status_text = view.status.value
    if not findings:
        summary = f"{status_text}; no problems detected"
    else:
        worst = next((s for s in ("error", "warning", "info") if any(f["severity"] == s for f in findings)), "info")
        summary = f"{status_text}; {len(findings)} finding(s), worst: {worst}"
    return {"run_id": store.run_id, "status": status_text, "metrics": view.metrics,
            "tunable_hyperparameters": sorted(tunable), "findings": findings, "summary": summary}
