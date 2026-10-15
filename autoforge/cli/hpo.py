"""`autoforge hpo ...`: hyperparameter search."""

from __future__ import annotations

import time
from pathlib import Path

import typer
import yaml

from autoforge.cli.output import JsonOpt, emit, errors_as_envelope
from autoforge.cli.runs import ActorOpt, _actor

hpo_app = typer.Typer(help="Hyperparameter search (each trial is a normal run).", no_args_is_help=True)


def _line(v: dict) -> str:
    best = "-" if v["best_value"] is None else f"{v['best_value']:.4g}"
    base = "-" if v["baseline_value"] is None else f"{v['baseline_value']:.4g}"
    imp = "" if v["improvement"] is None else f" ({v['improvement']:+.4g} vs baseline)"
    return (f"{v['study_id']:<34} {v['status']:<10} {v['completed']}/{v['n_trials']} trials, "
            f"{v['metric']} best={best} baseline={base}{imp}")


@hpo_app.command("start")
def start(
    plan_file: Path,
    metric: str = typer.Option(..., "--metric", "-m", help="Run metric to optimise, e.g. f1_macro or rmse."),
    trials: int = typer.Option(20, "--trials", "-n"),
    direction: str = typer.Option(None, "--direction", help="maximize / minimize (default: inferred from the metric)."),
    space_file: Path = typer.Option(None, "--space", help="YAML search space (default: the recipe's search_space)."),
    timeout_min: float = typer.Option(None, "--timeout-min", help="Stop starting new trials after N minutes."),
    wait: bool = typer.Option(False, "--wait", help="Block and print each finished trial."),
    actor: str = ActorOpt,
    as_json: bool = JsonOpt,
) -> None:
    """Start a search. Trial 0 is the plan's own hyperparameters, the baseline."""
    from autoforge import api
    from autoforge.policy.privacy import brain_from_actor

    space = yaml.safe_load(space_file.read_text(encoding="utf-8")) if space_file else None
    who = _actor(actor)
    with errors_as_envelope(as_json):
        plan = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
        view = api.hpo_start(plan, metric, trials, who, brain_from_actor(who), direction, space, timeout_min,
                             base=plan_file.resolve().parent)
    sid = view["study_id"]
    if not wait:
        emit(as_json, view, f"Started {sid}\nPoll with: autoforge hpo status {sid}")
        return
    if not as_json:
        typer.echo(f"Started {sid}")
    seen = 0
    while True:
        view = api.hpo_status(sid)
        if not as_json:
            for t in view["trials"][seen:]:
                tag = " (baseline)" if t.get("baseline") else ""
                res = f"{metric}={t['value']:.4g}" if "value" in t else f"failed: {t.get('error')}"
                typer.echo(f"  trial {t['number']}{tag}: {res}  {t['params']}")
        seen = len(view["trials"])
        if view["status"] not in ("pending", "running"):
            break
        time.sleep(1)
    emit(as_json, view, _line(view) + (f"\nbest params: {view['best_params']}  run: {view['best_run_id']}"
                                       if view["best_params"] is not None else ""))


@hpo_app.command("status")
def status(study_id: str, as_json: bool = JsonOpt) -> None:
    """Progress, best result and every trial."""
    from autoforge import api

    with errors_as_envelope(as_json):
        v = api.hpo_status(study_id)
    lines = [_line(v)] + [
        f"  trial {t['number']}{' (baseline)' if t.get('baseline') else ''}: "
        f"{t.get('value', 'failed')}  {t['params']}" for t in v["trials"]]
    emit(as_json, v, "\n".join(lines))


@hpo_app.command("list")
def list_cmd(as_json: bool = JsonOpt) -> None:
    """Recent searches."""
    from autoforge import api

    rows = api.hpo_list()
    emit(as_json, rows, "\n".join(map(_line, rows)) or "(no searches)")


@hpo_app.command("stop")
def stop(study_id: str, actor: str = ActorOpt, as_json: bool = JsonOpt) -> None:
    """Stop a search and its current trial; finished trials are kept."""
    from autoforge import api

    with errors_as_envelope(as_json):
        v = api.hpo_stop(study_id, _actor(actor))
    emit(as_json, v, _line(v))
