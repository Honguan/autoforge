"""`autoforge data ...`: mark dataset sensitivity (humans only) and profile data."""

from __future__ import annotations

from pathlib import Path

import typer

from autoforge.cli.output import JsonOpt, emit, errors_as_envelope
from autoforge.data import registry

data_app = typer.Typer(help="Dataset sensitivity and profiling.", no_args_is_help=True)


@data_app.command("mark")
def mark(
    path: Path,
    sensitivity: registry.Sensitivity = typer.Option(..., "--sensitivity", "-s",
                                                     help="public / internal / sensitive"),
    as_json: bool = JsonOpt,
) -> None:
    """Mark a file or directory. Sensitive data can only be used by a human or a local-model AI."""
    key = registry.mark(path, sensitivity)
    emit(as_json, {"path": key, "sensitivity": sensitivity.value}, f"{key}: {sensitivity.value}")


@data_app.command("list")
def list_marks(as_json: bool = JsonOpt) -> None:
    """All marked paths (unmarked data is 'internal')."""
    marks = registry.all_marks()
    emit(as_json, marks, "\n".join(f"{v:<10} {k}" for k, v in marks.items()) or "(nothing marked; default: internal)")


@data_app.command("profile")
def profile(path: Path, as_json: bool = JsonOpt) -> None:
    """Profile a dataset (as a human: full access, including sample rows)."""
    from autoforge import api
    from autoforge.schemas.core import BrainKind

    with errors_as_envelope(as_json):
        data = api.data_profile(str(path), BrainKind.human)
    emit(as_json, data)
