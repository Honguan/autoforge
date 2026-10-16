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


GPU_MARK = {"pass": "✅", "fail": "❌", "skip": "⏭️", "error": "⚠️"}


@eval_app.command("gpu")
def gpu(
    full: bool = typer.Option(False, "--full", help="Production-size models (QLoRA 1.5B, whisper-small, SD 1.5 at 512 px)."),
    check: list[str] = typer.Option(None, "--check", "-c", help="Run only these check ids (repeatable)."),
    packs_dir: str = typer.Option(None, "--packs-dir", help="Install packs from this folder (a repo checkout) instead of the index."),
    timeout: float = typer.Option(3600, "--timeout", help="Seconds allowed per training run."),
    list_checks: bool = typer.Option(False, "--list", help="List the checks and exit."),
    device: str = typer.Option("cuda", "--device", hidden=True, help="cpu only to test this command itself."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable report."),
) -> None:
    """Train each GPU pack for real on this NVIDIA GPU (small data), measure VRAM, register and predict.

    quick (default): small real models, a few minutes per check. --full: the recipes' real sizes.
    """
    from pathlib import Path

    from autoforge.evals.gpu_check import CHECKS, by_id, run_checks
    from autoforge.hardware.detect import detect_hardware

    if list_checks:
        rows = [{"id": c.id, "pack": c.pack, "recipe": c.recipe, "note": c.note} for c in CHECKS]
        emit(as_json, rows, "\n".join(f"{r['id']:22} {r['pack']}/{r['recipe']}  {r['note']}" for r in rows))
        return
    try:
        selected = by_id(check)
    except ValueError as e:
        fail(as_json, "BAD_REQUEST", str(e))
    if device == "cuda" and detect_hardware().best_device != "cuda":
        fail(as_json, "NO_GPU", "no usable NVIDIA GPU with CUDA was found",
             hint='run `autoforge doctor`; install the CUDA build of PyTorch: pip install "autoforge-trainer[deep]"')
    if not as_json:
        typer.echo(f"Checking {len(selected)} GPU pack(s), {'full' if full else 'quick'} tier "
                   "(downloads pretrained models the first time).")

    def progress(r) -> None:
        if not as_json:
            typer.echo(f"  {GPU_MARK[r.verdict]} {r.id:22} {r.seconds:6.0f} s  VRAM {r.peak_vram_gb:5.2f} GB  {r.reason}")

    with errors_as_envelope(as_json):
        report = run_checks(selected, "full" if full else "quick", device, Path(packs_dir).resolve() if packs_dir else None,
                            timeout, on_progress=progress)
    c = report["counts"]
    human = (f"\n{c['pass']} passed, {c['fail']} failed, {c['error']} errors, {c['skip']} skipped"
             f"\nreport: {report['report_dir']}/report.md")
    emit(as_json, report, human)
    if not report["all_passed"]:
        raise typer.Exit(1)
