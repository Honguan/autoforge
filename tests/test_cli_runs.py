import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autoforge.cli.main import app

REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()


def call(*args):
    r = runner.invoke(app, [*args, "--json"])
    return r.exit_code, json.loads(r.stdout)


def test_example_churn_plan_end_to_end():
    pytest.importorskip("sklearn")
    assert call("pack", "install", str(REPO / "packs" / "tabular-classic"))[0] == 0
    code, body = call("plan", "validate", str(REPO / "examples" / "churn_plan.yaml"))
    assert code == 0 and body["data"]["device"] == "cpu"
    code, body = call("run", "start", str(REPO / "examples" / "churn_plan.yaml"), "--wait", "--timeout", "120")
    assert code == 0, body
    view = body["data"]
    assert view["status"] == "succeeded", view
    assert view["metrics"]["f1_macro"] > 0.5
    assert Path(view["model_path"]).is_file()
    run_id = view["run_id"]
    assert call("run", "metrics", run_id)[1]["data"]["accuracy"]["count"] == 1
    assert [r["action"] for r in call("run", "audit", run_id)[1]["data"]] == ["create", "start"]
    assert call("run", "list")[1]["data"][0]["run_id"] == run_id
    code, body = call("run", "pause", run_id)
    assert code == 1 and body["error"]["code"] == "RUN_STATE_ERROR"


def test_invalid_plan_returns_problems(tmp_path):
    (tmp_path / "p.yaml").write_text(
        "task: {task_type: t, input_modality: x, output_modality: y, metric: m}\n"
        "data_path: missing.csv\npack: not-installed\nrecipe: r\n")
    code, body = call("run", "start", str(tmp_path / "p.yaml"))
    assert code == 1 and body["error"]["code"] == "PLAN_INVALID"
    probs = body["data"]["problems"]
    assert any("data_path not found" in p for p in probs) and any("not-installed" in p for p in probs)


def test_run_status_unknown():
    code, body = call("run", "status", "nope")
    assert code == 1 and body["error"]["code"] == "RUN_NOT_FOUND"
