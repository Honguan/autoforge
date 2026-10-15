import json

from typer.testing import CliRunner

from autoforge.cli.main import app

runner = CliRunner()


def test_doctor_json_envelope():
    r = runner.invoke(app, ["doctor", "--json"])
    assert r.exit_code == 0
    body = json.loads(r.stdout)
    assert body["ok"] is True
    assert body["data"]["best_device"] in ("cpu", "cuda")


def test_schema_unknown_returns_error_envelope():
    r = runner.invoke(app, ["schema", "Nope", "--json"])
    assert r.exit_code == 1
    body = json.loads(r.stdout)
    assert body["ok"] is False and body["error"]["code"] == "UNKNOWN_SCHEMA"


def test_schema_training_plan():
    r = runner.invoke(app, ["schema", "TrainingPlan", "--json"])
    assert "hyperparameters" in json.loads(r.stdout)["data"]["properties"]
