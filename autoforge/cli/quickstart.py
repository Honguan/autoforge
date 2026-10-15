"""`autoforge quickstart`: the whole loop in one command, on bundled example data."""

from __future__ import annotations

import time
from importlib import resources

import typer

from autoforge.cli.output import JsonOpt, emit, errors_as_envelope


def quickstart(
    recipe: str = typer.Option("random_forest", "--recipe", help="logistic_regression or random_forest"),
    as_json: bool = JsonOpt,
) -> None:
    """Hardware check → install a pack → train → register → predict, on a bundled churn dataset (~1 min)."""
    from autoforge import api
    from autoforge.errors import AutoForgeError
    from autoforge.packs import installer
    from autoforge.schemas.core import BrainKind

    start = time.monotonic()
    log = (lambda *a: None) if as_json else typer.echo
    data = str(resources.files("autoforge.examples") / "churn.csv")
    with errors_as_envelope(as_json):
        hw = api.hardware()
        log(f"1/5 hardware: {hw['cpu']['model']}, device {hw['best_device']}")
        if "tabular-classic" not in installer.read_lock():
            try:
                rec = api.pack_install("tabular-classic")
            except AutoForgeError:  # offline or index unreachable: fall back to the GitHub source
                rec = api.pack_install("github:Honguan/autoforge/packs/tabular-classic@main")
            log(f"2/5 installed pack tabular-classic {rec['version']}")
        else:
            log("2/5 pack tabular-classic already installed")
        plan = {"task": {"task_type": "tabular-classification", "input_modality": "tabular", "output_modality": "label",
                         "target": "churned", "metric": "f1_macro"},
                "data_path": data, "pack": "tabular-classic", "recipe": recipe,
                "hyperparameters": {"class_weight": "balanced"}}
        run = api.run_start(plan, "human", BrainKind.human)
        log(f"3/5 training {run['run_id']} …")
        view = api.run_wait(run["run_id"], 120)
        if view["status"] != "succeeded":
            raise AutoForgeError(f"training {view['status']}: {view.get('error')}",
                                 hint=f"autoforge run logs {run['run_id']}")
        log("    " + ", ".join(f"{k}={v:.3f}" for k, v in view["metrics"].items()))
        card = api.model_register(run["run_id"], "quickstart-churn", "human", "Quickstart churn model")
        log(f"4/5 registered model {card['name']}:v{card['version']}")
        rows = [{"tenure_months": 2, "monthly_fee": 110, "contract": "month-to-month", "support_calls": 5},
                {"tenure_months": 60, "monthly_fee": 25, "contract": "two-year", "support_calls": 0}]
        pred = api.model_predict(card["name"], rows)
        for r, p in zip(rows, pred["predictions"]):
            log(f"5/5 predict {r} → churn={p['prediction']} (p={p['probabilities'].get('1', 0):.2f})")
    seconds = round(time.monotonic() - start, 1)
    emit(as_json, {"seconds": seconds, "run_id": run["run_id"], "model": f"{card['name']}:v{card['version']}",
                   "metrics": view["metrics"], "predictions": pred["predictions"]},
         f"\nDone in {seconds} s. Next:\n"
         f"  autoforge gui                      # web interface\n"
         f"  autoforge serve                    # REST API for the model\n"
         f"  claude mcp add autoforge -- autoforge mcp   # let Claude drive AutoForge")
