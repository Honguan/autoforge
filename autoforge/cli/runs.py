"""`autoforge run ...` and `autoforge plan ...` commands."""

from __future__ import annotations

import os
import time
from pathlib import Path

import typer

from autoforge.cli.output import JsonOpt, emit, errors_as_envelope, fail
from autoforge.engine import runner
from autoforge.engine.store import TERMINAL, RunStore
from autoforge.engine.validate import check_plan, load_plan_file

run_app = typer.Typer(help="Start and control training runs.", no_args_is_help=True)
plan_app = typer.Typer(help="Check training plans.", no_args_is_help=True)

ActorOpt = typer.Option(
    None, "--actor", help="Who is acting: 'human' or 'ai:<name>'. Defaults to $AUTOFORGE_ACTOR or 'human'."
)


def _actor(actor: str | None) -> str:
    return actor or os.environ.get("AUTOFORGE_ACTOR", "human")


def _view_line(v: runner.RunView) -> str:
    m = v.metrics or v.latest_metrics
    shown = ", ".join(f"{k}={val:.4g}" for k, val in m.items())
    return f"{v.run_id:<40} {v.status.value:<10} {v.recipe or 'custom':<20} {shown}"


def _follow(store: RunStore, timeout: float | None) -> runner.RunView:
    """Print metric/log events until the run finishes (human mode for --wait)."""
    seen = 0
    end = time.monotonic() + timeout if timeout else None
    while True:
        events = store.events()
        for e in events[seen:]:
            if e.get("type") == "metric":
                step = f"step {e['step']}: " if e.get("step") is not None else ""
                typer.echo(f"  {step}{e['name']} = {e['value']:.4g}")
            elif e.get("type") == "log":
                typer.echo(f"  {e['message']}")
        seen = len(events)
        view = runner.status(store)
        if view.status in TERMINAL or (end and time.monotonic() > end):
            return view
        time.sleep(0.3)


@plan_app.command("validate")
def plan_validate(plan_file: Path, as_json: bool = JsonOpt) -> None:
    """Check a plan against installed packs, data and hardware. Nothing is run."""
    with errors_as_envelope(as_json):
        plan = load_plan_file(plan_file)
    check = check_plan(plan)
    if not check.ok:
        fail(as_json, "PLAN_INVALID", f"{len(check.problems)} problem(s) in plan",
             "Fix every item in details.problems.", check.model_dump())
    human = "\n".join([f"OK: device={check.device}, entrypoint={check.entrypoint}",
                       *(f"warning: {w}" for w in check.warnings)])
    emit(as_json, check.model_dump(), human)


@run_app.command("start")
def start(
    plan_file: Path,
    wait: bool = typer.Option(False, "--wait", help="Block until the run finishes."),
    timeout: float = typer.Option(None, "--timeout", help="Max seconds to wait with --wait."),
    actor: str = ActorOpt,
    as_json: bool = JsonOpt,
) -> None:
    """Validate a plan and start training in the background. Returns the run_id immediately."""
    with errors_as_envelope(as_json):
        plan = load_plan_file(plan_file)
        store = runner.create_run(plan, _actor(actor))
        runner.start(store, _actor(actor))
    if not wait:
        view = runner.status(store)
        emit(as_json, view.model_dump(mode="json"), f"Started {store.run_id}\nPoll with: autoforge run status {store.run_id}")
        return
    if as_json:
        view = runner.wait(store, timeout)
    else:
        typer.echo(f"Started {store.run_id}")
        view = _follow(store, timeout)
    emit(as_json, view.model_dump(mode="json"), _view_line(view) + (f"\nerror: {view.error}" if view.error else ""))


@run_app.command("status")
def status(run_id: str, as_json: bool = JsonOpt) -> None:
    """Current status, final metrics and latest logged metrics of a run."""
    with errors_as_envelope(as_json):
        view = runner.status(RunStore.open(run_id))
    emit(as_json, view.model_dump(mode="json"), _view_line(view) + (f"\nerror: {view.error}" if view.error else ""))


@run_app.command("list")
def list_cmd(as_json: bool = JsonOpt) -> None:
    """All runs, newest first."""
    views = runner.all_runs()
    emit(as_json, [v.model_dump(mode="json") for v in views], "\n".join(map(_view_line, views)) or "(no runs)")


def _control(action: str, run_id: str, actor: str | None, as_json: bool, **kw) -> None:
    with errors_as_envelope(as_json):
        store = RunStore.open(run_id)
        view = getattr(runner, action)(store, _actor(actor), **kw)
    emit(as_json, view.model_dump(mode="json"), _view_line(view))


@run_app.command("pause")
def pause(run_id: str, actor: str = ActorOpt, as_json: bool = JsonOpt) -> None:
    """Suspend a running run (works with any trainer)."""
    _control("pause", run_id, actor, as_json)


@run_app.command("resume")
def resume(run_id: str, actor: str = ActorOpt, as_json: bool = JsonOpt) -> None:
    """Continue a paused run."""
    _control("resume", run_id, actor, as_json)


@run_app.command("stop")
def stop(
    run_id: str,
    grace: float = typer.Option(10.0, "--grace", help="Seconds to wait for a clean stop before terminating."),
    actor: str = ActorOpt,
    as_json: bool = JsonOpt,
) -> None:
    """Ask the run to stop; terminate it if it does not stop within the grace period."""
    _control("stop", run_id, actor, as_json, grace_seconds=grace)


@run_app.command("retry")
def retry(
    run_id: str,
    wait: bool = typer.Option(False, "--wait"),
    actor: str = ActorOpt,
    as_json: bool = JsonOpt,
) -> None:
    """Start a new run with the same plan, resuming from the old run's checkpoints if any."""
    with errors_as_envelope(as_json):
        store = runner.retry(RunStore.open(run_id), _actor(actor))
        runner.start(store, _actor(actor))
    view = runner.wait(store) if wait else runner.status(store)
    emit(as_json, view.model_dump(mode="json"), f"Started {store.run_id} (retry of {run_id})")


@run_app.command("logs")
def logs(run_id: str, tail: int = typer.Option(50, "--tail"), as_json: bool = JsonOpt) -> None:
    """Last lines of the worker's stdout/stderr."""
    with errors_as_envelope(as_json):
        lines = RunStore.open(run_id).logs_tail(tail)
    emit(as_json, {"lines": lines}, "\n".join(lines))


@run_app.command("metrics")
def metrics(run_id: str, last: int = typer.Option(5, "--last"), as_json: bool = JsonOpt) -> None:
    """Per-metric summary: last, min, max, count and the most recent values."""
    with errors_as_envelope(as_json):
        summary = runner.metrics_summary(RunStore.open(run_id), last)
    human = "\n".join(
        f"{k:<20} last={v['last']:.4g} min={v['min']:.4g} max={v['max']:.4g} n={v['count']}"
        for k, v in summary.items()
    ) or "(no metrics yet)"
    emit(as_json, summary, human)


@run_app.command("audit")
def audit(run_id: str, as_json: bool = JsonOpt) -> None:
    """Every action taken on a run and who took it."""
    with errors_as_envelope(as_json):
        rows = RunStore.open(run_id).audit_log()
    emit(as_json, rows, "\n".join(f"{r['t']}  {r['actor']:<16} {r['action']}" for r in rows))
