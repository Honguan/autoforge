"""`autoforge eval agent`: measure how often a brain chooses a reasonable training method."""

from __future__ import annotations

import typer

from autoforge.agents.providers import make_provider, parse_brain
from autoforge.cli.output import emit, errors_as_envelope, fail

eval_app = typer.Typer(help="Evaluate the built-in agent.", no_args_is_help=True)

MARK = {"pass": "✅", "partial": "🟡", "fail": "❌", "error": "⚠️"}


@eval_app.command("agent")
def agent(
    brain: str = typer.Option(None, "--brain", "-b", help="provider:model (default $AUTOFORGE_BRAIN or anthropic:claude-opus-5-5)."),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible endpoint for openai:/local:/ollama:."),
    api_key_env: str = typer.Option(None, "--api-key-env", help="Env var holding the API key for openai:."),
    case: list[str] = typer.Option(None, "--case", "-c", help="Run only these case ids (repeatable)."),
    max_steps: int = typer.Option(25, "--max-steps", help="Model turns per case."),
    list_cases: bool = typer.Option(False, "--list", help="List the cases and exit."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable report."),
) -> None:
    """Give the agent each case's request and data; score the method it proposes (nothing is trained).

    Every case costs one agent conversation (about 5–20 model calls) with the chosen brain.
    """
    from autoforge.evals.agent_eval import run_eval
    from autoforge.evals.cases import CASES, by_id

    if list_cases:
        rows = [{"id": c.id, "goal": c.goal, "expected": sorted(f"{p}/{r}" for p, r in c.methods),
                 "expect": c.expect, "note": c.note} for c in CASES]
        emit(as_json, rows, "\n".join(f"{r['id']:20} {r['goal']}" for r in rows))
        return
    with errors_as_envelope(as_json):
        spec = parse_brain(brain, base_url, api_key_env)
        make_provider(spec)  # fail fast (missing key / package) before writing any data
        try:
            selected = by_id(case)
        except ValueError as e:
            fail(as_json, "BAD_REQUEST", str(e))
    if not as_json:
        typer.echo(f"Evaluating {spec.provider}:{spec.model} on {len(selected)} case(s); nothing will be trained.")

    def progress(r) -> None:
        if not as_json:
            chose = f"{r.pack}/{r.recipe}" if r.pack else "—"
            typer.echo(f"  {MARK[r.verdict]} {r.id:20} {chose:42} {r.reason}")

    with errors_as_envelope(as_json):
        report = run_eval(selected, spec, make_provider, max_steps=max_steps, on_progress=progress)
    c = report["counts"]
    human = (f"\n{c['pass']}/{report['total']} passed ({report['pass_rate']:.0%}), {c['partial']} partial, "
             f"{c['fail']} failed, {c['error']} errors — target {'met' if report['met_target'] else 'NOT met'}"
             f"\nreport: {report['report_dir']}/report.md")
    emit(as_json, report, human)
