"""`autoforge train "<goal>"`: the built-in agent plans and trains from one sentence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from autoforge.agents.loop import AgentEvent, describe_call, run_agent
from autoforge.agents.providers import ToolCall, make_provider, parse_brain
from autoforge.cli.output import emit, errors_as_envelope, fail

_SHORT = {"hardware_profile", "pack_list", "run_status", "run_wait", "run_metrics", "run_list"}


async def _ask(call: ToolCall) -> tuple[bool, str]:
    typer.echo("\n" + typer.style("The agent wants to start training:", bold=True))
    typer.echo(describe_call(call))
    answer = typer.prompt("Start? [y = yes / n = no / or type feedback for the agent]", default="n").strip()
    if answer.lower() in ("y", "yes"):
        return True, ""
    return False, "" if answer.lower() in ("n", "no") else answer


def _printer(event: AgentEvent) -> None:
    d = event.data
    if event.kind == "text":
        typer.echo("\n" + d["text"])
    elif event.kind == "tool_call":
        args = "" if event.data["name"] in _SHORT else json.dumps(d["arguments"], ensure_ascii=False)[:160]
        typer.echo(typer.style(f"  → {d['name']} {args}", dim=True))
    elif event.kind == "tool_result" and not d["ok"]:
        typer.echo(typer.style(f"    ✗ {d['name']}: {d['error']}", fg="yellow"))
    elif event.kind == "info":
        typer.echo(typer.style(d["message"], fg="yellow"))


def train(
    goal: str = typer.Argument(help="What you want, in plain words. e.g. 'predict churned from this CSV'"),
    data: Path = typer.Option(None, "--data", "-d", help="Dataset file or directory."),
    brain: str = typer.Option(
        None, "--brain", "-b",
        help="provider:model. anthropic:claude-opus-5-5 (default), openai:<model>, ollama:<model>, local:<model>. "
             "Default from $AUTOFORGE_BRAIN.",
    ),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible endpoint for openai:/local:/ollama:."),
    api_key_env: str = typer.Option(None, "--api-key-env", help="Env var holding the API key for openai: (default OPENAI_API_KEY)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before starting training."),
    max_steps: int = typer.Option(40, "--max-steps", help="Maximum model turns."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable result (requires --yes)."),
) -> None:
    """Let an AI analyse your data and hardware, choose a method, train and report — from one sentence."""
    if as_json and not yes:
        fail(True, "BAD_REQUEST", "--json needs --yes (no interactive approval possible)")
    with errors_as_envelope(as_json):
        spec = parse_brain(brain, base_url, api_key_env)
        provider = make_provider(spec)
    if not as_json:
        where = "local model" if spec.is_local else "cloud model"
        typer.echo(typer.style(f"Brain: {spec.provider}:{spec.model} ({where}) — actor {spec.actor}", dim=True))
    with errors_as_envelope(as_json):
        result = asyncio.run(run_agent(
            goal,
            provider,
            actor=spec.actor,
            brain=spec.brain_kind,
            approve=None if yes else _ask,
            on_event=None if as_json else _printer,
            max_steps=max_steps,
            data_path=str(data) if data else None,
        ))
    data_out = {"stop": result.stop, "steps": result.steps, "runs": result.runs,
                "final_text": result.final_text, "transcript": result.transcript}
    human = (f"\n— {result.stop} after {result.steps} step(s); runs: {', '.join(result.runs) or 'none'}"
             f"\n  transcript: {result.transcript}")
    emit(as_json, data_out, human)
    if result.stop not in ("end",):
        raise typer.Exit(1)
