"""Shared output helpers: every command emits the Envelope shape with --json."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

import typer

from autoforge.packs.errors import PackError, PackInvalid
from autoforge.schemas.core import Envelope, ErrorInfo

JsonOpt = typer.Option(False, "--json", help="Machine-readable JSON output.")


def emit(as_json: bool, data: Any = None, human: str | None = None) -> None:
    if as_json:
        typer.echo(Envelope(ok=True, data=data).model_dump_json(indent=2))
    else:
        typer.echo(human if human is not None else json.dumps(data, indent=2, ensure_ascii=False))


def fail(as_json: bool, code: str, message: str, hint: str | None = None, details: Any = None) -> None:
    if as_json:
        err = ErrorInfo(code=code, message=message, hint=hint)
        env = Envelope(ok=False, error=err, data=details)
        typer.echo(env.model_dump_json(indent=2))
    else:
        typer.echo(f"Error [{code}]: {message}" + (f"\nHint: {hint}" if hint else ""), err=True)
    raise typer.Exit(1)


@contextmanager
def errors_as_envelope(as_json: bool) -> Iterator[None]:
    """Turn PackError (and subclasses) into a JSON error envelope with its code."""
    try:
        yield
    except PackInvalid as e:
        fail(as_json, e.code, "pack is invalid", e.hint, {"problems": e.problems})
    except PackError as e:
        fail(as_json, e.code, str(e), e.hint)
