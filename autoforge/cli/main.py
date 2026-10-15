"""AutoForge CLI. Every command supports --json and emits the Envelope shape,
so AI agents can call it as reliably as the MCP server."""

from __future__ import annotations

import json
from typing import Any

import typer

from autoforge import __version__
from autoforge.hardware.detect import detect_hardware
from autoforge.schemas.core import SCHEMAS, Envelope, ErrorInfo

app = typer.Typer(help="AutoForge: AI that trains AI.", no_args_is_help=True)

JsonOpt = typer.Option(False, "--json", help="Machine-readable JSON output.")


def _emit(as_json: bool, data: Any = None, human: str | None = None) -> None:
    if as_json:
        typer.echo(Envelope(ok=True, data=data).model_dump_json(indent=2))
    else:
        typer.echo(human if human is not None else json.dumps(data, indent=2, ensure_ascii=False))


def _fail(as_json: bool, code: str, message: str, hint: str | None = None) -> None:
    if as_json:
        err = ErrorInfo(code=code, message=message, hint=hint)
        typer.echo(Envelope(ok=False, error=err).model_dump_json(indent=2))
    else:
        typer.echo(f"Error [{code}]: {message}" + (f"\nHint: {hint}" if hint else ""), err=True)
    raise typer.Exit(1)


@app.command()
def version(as_json: bool = JsonOpt) -> None:
    """Show the AutoForge version."""
    _emit(as_json, {"version": __version__}, f"autoforge {__version__}")


@app.command()
def doctor(as_json: bool = JsonOpt) -> None:
    """Detect CPU / RAM / NVIDIA GPU / CUDA and report the training device."""
    hw = detect_hardware()
    data = hw.model_dump() | {"best_device": hw.best_device}
    if as_json:
        _emit(True, data)
        return
    lines = [
        f"OS:      {hw.os} (Python {hw.python_version})",
        f"CPU:     {hw.cpu.model} ({hw.cpu.physical_cores}C/{hw.cpu.logical_cores}T)",
        f"RAM:     {hw.ram_available_gb} / {hw.ram_total_gb} GB available",
        f"Disk:    {hw.disk_free_gb} GB free",
    ]
    if hw.gpus:
        for g in hw.gpus:
            lines.append(
                f"GPU {g.index}:   {g.name}, {g.vram_total_gb} GB VRAM, "
                f"CC {g.compute_capability}, driver {g.driver_version}"
            )
        lines.append(f"CUDA:    {hw.cuda_version}")
    else:
        lines.append("GPU:     none (NVIDIA)")
    lines.append(f"Torch:   {hw.torch_version or 'not installed'} (CUDA usable: {hw.torch_cuda_available})")
    lines.append(f"Device:  {hw.best_device}")
    for n in hw.detection_notes:
        lines.append(f"  note: {n}")
    _emit(False, human="\n".join(lines))


@app.command()
def schema(
    name: str = typer.Argument(None, help="Schema name; omit to list all."),
    as_json: bool = JsonOpt,
) -> None:
    """Print the JSON Schema of a core data structure."""
    if name is None:
        _emit(as_json, sorted(SCHEMAS), "\n".join(sorted(SCHEMAS)))
        return
    if name not in SCHEMAS:
        _fail(as_json, "UNKNOWN_SCHEMA", f"No schema named '{name}'", f"Choose from: {', '.join(sorted(SCHEMAS))}")
    _emit(as_json, SCHEMAS[name].model_json_schema())


if __name__ == "__main__":
    app()
