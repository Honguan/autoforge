"""AutoForge CLI. Every command supports --json and emits the Envelope shape,
so AI agents can call it as reliably as the MCP server."""

from __future__ import annotations

import typer

from autoforge import __version__
from autoforge.cli.output import JsonOpt, emit, fail
from autoforge.cli.packs import index_app, pack_app
from autoforge.cli.agent import train
from autoforge.cli.data import data_app
from autoforge.cli.hpo import hpo_app
from autoforge.cli.models import gui, model_app, serve, serve_mcp
from autoforge.cli.runs import plan_app, run_app
from autoforge.hardware.detect import detect_hardware
from autoforge.engine.runner import RunView
from autoforge.registry.models import ModelCard
from autoforge.packs.index import IndexEntry, PackIndex
from autoforge.packs.manifest import PackManifest, RecipeSpec
from autoforge.schemas.core import SCHEMAS

app = typer.Typer(help="AutoForge: AI that trains AI.", no_args_is_help=True)
app.add_typer(pack_app, name="pack")
app.add_typer(index_app, name="index")
app.add_typer(plan_app, name="plan")
app.add_typer(run_app, name="run")
app.add_typer(data_app, name="data")
app.add_typer(hpo_app, name="hpo")
app.add_typer(model_app, name="model")
app.command()(serve)
app.command("serve-mcp")(serve_mcp)
app.command()(train)
app.command()(gui)

ALL_SCHEMAS = SCHEMAS | {
    "PackManifest": PackManifest,
    "RecipeSpec": RecipeSpec,
    "PackIndex": PackIndex,
    "IndexEntry": IndexEntry,
    "RunView": RunView,
    "ModelCard": ModelCard,
}


@app.command()
def version(as_json: bool = JsonOpt) -> None:
    """Show the AutoForge version."""
    emit(as_json, {"version": __version__}, f"autoforge {__version__}")


@app.command()
def doctor(as_json: bool = JsonOpt) -> None:
    """Detect CPU / RAM / NVIDIA GPU / CUDA and report the training device."""
    hw = detect_hardware()
    data = hw.model_dump() | {"best_device": hw.best_device}
    if as_json:
        emit(True, data)
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
    emit(False, human="\n".join(lines))


@app.command()
def mcp(
    actor: str = typer.Option(
        "ai:mcp", "--actor",
        help="Recorded in audit logs. Use 'ai:local:<model>' only if the MCP client is driven by a local model.",
    ),
) -> None:
    """Run the MCP server over stdio (for Claude Desktop, Claude Code and other MCP clients)."""
    from autoforge.serve.mcp_server import main as serve

    serve(actor)


@app.command()
def schema(
    name: str = typer.Argument(None, help="Schema name; omit to list all."),
    as_json: bool = JsonOpt,
) -> None:
    """Print the JSON Schema of a core data structure."""
    if name is None:
        emit(as_json, sorted(ALL_SCHEMAS), "\n".join(sorted(ALL_SCHEMAS)))
        return
    if name not in ALL_SCHEMAS:
        fail(as_json, "UNKNOWN_SCHEMA", f"No schema named '{name}'", f"Choose from: {', '.join(sorted(ALL_SCHEMAS))}")
    emit(as_json, ALL_SCHEMAS[name].model_json_schema())


if __name__ == "__main__":
    app()
