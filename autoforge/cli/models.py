"""`autoforge model ...`, `autoforge serve` and `autoforge serve-mcp`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from autoforge.cli.output import JsonOpt, emit, errors_as_envelope, fail
from autoforge.cli.runs import ActorOpt, _actor

model_app = typer.Typer(help="Registered models: register, inspect, predict, export.", no_args_is_help=True)


@model_app.command("register")
def register(
    run_id: str,
    name: str = typer.Option(..., "--name", "-n", help="Model name (lowercase, digits, '-', '_')."),
    description: str = typer.Option("", "--description", "-d"),
    actor: str = ActorOpt,
    as_json: bool = JsonOpt,
) -> None:
    """Turn a succeeded run into the next version of a model."""
    from autoforge import api

    with errors_as_envelope(as_json):
        card = api.model_register(run_id, name, _actor(actor), description)
    note = "" if card["servable"] else "\n(not servable: the recipe has no predictor)"
    emit(as_json, card, f"Registered {card['name']}:v{card['version']} from {run_id}{note}")


@model_app.command("list")
def list_cmd(as_json: bool = JsonOpt) -> None:
    """Registered models (latest version of each)."""
    from autoforge import api

    rows = api.model_list()
    human = "\n".join(
        f"{r['model']:<28} {r['task']:<24} " + ", ".join(f"{k}={v:.4g}" for k, v in r["metrics"].items())
        for r in rows) or "(no models)"
    emit(as_json, rows, human)


@model_app.command("info")
def info(model: str, as_json: bool = JsonOpt) -> None:
    """Model card: features, labels, metrics, source run."""
    from autoforge import api

    with errors_as_envelope(as_json):
        emit(as_json, api.model_info(model))


@model_app.command("predict")
def predict(
    model: str,
    rows_json: str = typer.Option(None, "--rows", help='JSON list of objects, e.g. \'[{"age": 30}]\''),
    rows_file: Path = typer.Option(None, "--file", "-f", help="JSON file, or CSV with a header row."),
    as_json: bool = JsonOpt,
) -> None:
    """Predict with a registered model."""
    import csv

    from autoforge import api

    if rows_file:
        if rows_file.suffix.lower() == ".csv":
            with rows_file.open(newline="", encoding="utf-8") as f:
                rows = [{k: v for k, v in r.items()} for r in csv.DictReader(f)]
        else:
            rows = json.loads(rows_file.read_text(encoding="utf-8"))
    elif rows_json:
        rows = json.loads(rows_json)
    else:
        fail(as_json, "BAD_REQUEST", "give --rows or --file")
    with errors_as_envelope(as_json):
        out = api.model_predict(model, rows)
    emit(as_json, out)


@model_app.command("export")
def export(
    model: str,
    fmt: str = typer.Option("bundle", "--format", help="bundle (zip) or a format the model supports, e.g. onnx."),
    out: Path = typer.Option(None, "--out", "-o"),
    as_json: bool = JsonOpt,
) -> None:
    """Export a model as a portable bundle or another format (e.g. ONNX)."""
    from autoforge import api

    with errors_as_envelope(as_json):
        res = api.model_export(model, fmt, str(out) if out else None)
    emit(as_json, res, f"Exported {res['model']} ({fmt}) → {res['path']}  sha256 {res['sha256'][:16]}…")


@model_app.command("import")
def import_cmd(bundle: Path, name: str = typer.Option(None, "--name"), actor: str = ActorOpt,
               as_json: bool = JsonOpt) -> None:
    """Import a bundle exported elsewhere as a new version."""
    from autoforge.registry import models as reg

    with errors_as_envelope(as_json):
        card = reg.import_bundle(bundle, name, _actor(actor))
    emit(as_json, card.model_dump(mode="json"), f"Imported {card.ref}")


@model_app.command("remove")
def remove(model: str, as_json: bool = JsonOpt) -> None:
    """Delete a model version (name:vN) or all versions (name)."""
    from autoforge.registry import models as reg

    with errors_as_envelope(as_json):
        gone = reg.remove(model)
    emit(as_json, gone, "Removed " + ", ".join(gone))


def serve(
    model: list[str] = typer.Option(None, "--model", "-m", help="Only serve these model names (repeatable)."),
    host: str = typer.Option("127.0.0.1", "--host", help="Use 0.0.0.0 to accept connections from other machines."),
    port: int = typer.Option(8000, "--port", "-p"),
    api_key_env: str = typer.Option(None, "--api-key-env", help="Env var holding an API key clients must send."),
) -> None:
    """Serve registered models over HTTP (REST)."""
    import uvicorn

    from autoforge.serve.rest import create_app

    key = None
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            fail(False, "BAD_REQUEST", f"{api_key_env} is not set")
    if host not in ("127.0.0.1", "localhost") and not key:
        typer.echo(typer.style("warning: serving on a network interface without an API key", fg="yellow"), err=True)
    typer.echo(f"Serving {'all models' if not model else ', '.join(model)} on http://{host}:{port}  (docs: docs/DEPLOY.md)")
    uvicorn.run(create_app(model or None, key), host=host, port=port, log_level="info")


def serve_mcp(
    model: list[str] = typer.Option(None, "--model", "-m", help="Only expose these model names (repeatable)."),
) -> None:
    """Expose registered models as MCP tools (predict_<name>) over stdio — prediction only, no training."""
    from autoforge.serve.model_mcp import main

    main(model or None)


def gui(
    port: int = typer.Option(8800, "--port", "-p"),
    host: str = typer.Option("127.0.0.1", "--host",
                             help="Bind address. Use 0.0.0.0 only inside Docker with -p 127.0.0.1:8800:8800."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the browser automatically."),
) -> None:
    """Open the web interface (local only: 127.0.0.1, protected by a one-time token)."""
    import secrets
    import threading
    import webbrowser

    import uvicorn

    from autoforge.gui.app import create_app

    token = secrets.token_urlsafe(24)
    url = f"http://127.0.0.1:{port}/"
    typer.echo(f"AutoForge GUI: {url}  (Ctrl+C to stop)")
    if host not in ("127.0.0.1", "localhost"):
        typer.echo(typer.style(f"warning: binding {host}; the GUI still only answers requests addressed to "
                               "localhost/127.0.0.1, so publish the port only on 127.0.0.1", fg="yellow"), err=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(token), host=host, port=port, log_level="warning")
