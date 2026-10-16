import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autoforge import api
from autoforge.cli.main import app
from autoforge.data import registry
from autoforge.engine import hpo, runner
from autoforge.engine.diagnose import _curve_findings
from autoforge.engine.store import RunStore, wait_for
from autoforge.engine.validate import PlanInvalid
from autoforge.packs import installer
from autoforge.policy.privacy import PrivacyDenied
from autoforge.schemas.core import BrainKind, RunStatus, Sensitivity, TaskSpec, TrainingPlan

pytest.importorskip("optuna")
FIXTURE = Path(__file__).parent / "fixtures" / "engine-pack"


@pytest.fixture
def toy(tmp_path):
    installer.install(str(FIXTURE))
    data = tmp_path / "data.txt"
    data.write_text("x")

    def make(recipe="quick", **kw):
        return TrainingPlan(task=TaskSpec(task_type="toy", input_modality="x", output_modality="y", metric="score"),
                            data_path=str(data), pack="engine-test", recipe=recipe, **kw)
    return make


def run(plan, actor="human"):
    s = runner.create_run(plan, actor)
    runner.start(s, actor)
    return s


# ---------------- hyperparameter search

def test_hpo_beats_baseline(toy):
    store = hpo.create_study(toy("score"), "score", n_trials=14, actor="ai:test")
    hpo.start_study(store)
    v = hpo.wait(store, 180)
    assert v.status == "succeeded", (v, (store.dir / "logs.txt").read_text()[-2000:])
    assert v.direction == "maximize" and v.completed == 14
    t0 = v.trials[0]
    assert t0["baseline"] and t0["params"] == {"x": 0, "lr": 0.01} and t0["value"] == 1.0
    assert v.baseline_value == 1.0 and v.best_value > v.baseline_value and v.improvement > 0
    best = RunStore.open(v.best_run_id).record()
    assert best.study_id == store.study_id and best.plan.hyperparameters["x"] == v.best_params["x"]
    assert {a["actor"] for a in RunStore.open(v.best_run_id).audit_log()} == {"ai:test"}


def test_hpo_minimize_direction_inferred(toy):
    store = hpo.create_study(toy("score"), "loss", n_trials=4)
    assert store.config().direction == "minimize"


def test_hpo_stop(toy):
    space = {"steps": {"choices": [100000]}}
    store = hpo.create_study(toy("slow"), "loss", n_trials=5, search_space=space)
    hpo.start_study(store)
    assert wait_for(lambda: hpo.status(store).current_run_id is not None, 60)
    v = hpo.stop(store)
    assert v.status == "stopped"
    assert all(r.status in ("stopped", "failed", "succeeded") for r in runner.all_runs())


def test_hpo_validation(toy):
    with pytest.raises(PlanInvalid) as e:
        hpo.create_study(toy("score"), "score", n_trials=1)
    assert "n_trials" in json.dumps(e.value.details)
    with pytest.raises(PlanInvalid):
        hpo.create_study(toy("score"), "score", search_space={"x": {"type": "int", "low": 5, "high": 1}})
    with pytest.raises(PlanInvalid):
        hpo.create_study(toy("quick"), "score")  # recipe declares no search space


def test_hpo_respects_privacy(toy, tmp_path):
    secret = tmp_path / "s.txt"
    secret.write_text("x")
    registry.mark(secret, Sensitivity.sensitive)
    with pytest.raises(PrivacyDenied):
        hpo.create_study(toy("score").model_copy(update={"data_path": str(secret)}), "score", actor="ai:claude")


# ---------------- diagnosis

def test_diagnose_failures(toy):
    s = run(toy("needs_missing_pkg"))
    runner.wait(s, 30)
    d = api.run_diagnose(s.run_id)
    assert d["findings"][0]["code"] == "MISSING_PACKAGE" and "autoforge_no_such_module" in d["findings"][0]["suggestion"]
    s = run(toy("crash"))
    runner.wait(s, 30)
    assert api.run_diagnose(s.run_id)["findings"][0]["code"] == "UNKNOWN_FAILURE"
    s = run(toy("quick"))
    runner.wait(s, 30)
    assert api.run_diagnose(s.run_id)["summary"].startswith("succeeded")


def test_curve_rules():
    f = _curve_findings({"loss": [1.0, 0.5, float("nan")]}, {"lr"})
    assert f[0]["code"] == "DIVERGED" and "lr" in f[0]["suggestion"]
    over = {"train_loss": [1.0, 0.8, 0.6, 0.4, 0.3, 0.2], "val_loss": [1.0, 0.7, 0.6, 0.65, 0.75, 0.9]}
    codes = [x["code"] for x in _curve_findings(over, {"dropout"})]
    assert "OVERFITTING" in codes
    plateau = {"loss": [1.0, 0.5, 0.40, 0.400, 0.399, 0.399, 0.399]}
    assert [x["code"] for x in _curve_findings(plateau, set())] == ["PLATEAU"]
    assert _curve_findings({"loss": [1.0, 0.8, 0.6, 0.4]}, set()) == []


# ---------------- live adjustment / comparison

def test_adjust_live_run(toy):
    s = run(toy("adjustable"), actor="ai:test")
    assert wait_for(lambda: s.state().status is RunStatus.running, 30)
    out = runner.adjust(s, "human", {"lr": 0.5})
    assert out["seq"] == 1
    assert wait_for(lambda: any(e.get("name") == "lr" and e["value"] == 0.5 for e in s.events()), 10)
    runner.adjust(s, "ai:test", {"finish": True})
    v = runner.wait(s, 30)
    assert v.status is RunStatus.succeeded and v.metrics["final_lr"] == 0.5
    assert [a["action"] for a in s.audit_log()].count("adjust") == 2
    log = subprocess.run(["git", "log", "--format=%s"], cwd=s.dir, capture_output=True, text=True).stdout
    assert "[human] adjust" in log and "[ai:test] adjust" in log
    with pytest.raises(runner.RunStateError):
        runner.adjust(s, "human", {"lr": 1})


def test_compare(toy):
    a, b = run(toy(hyperparameters={"lr": 0.1})), run(toy(hyperparameters={"lr": 0.5}))
    runner.wait(a, 30), runner.wait(b, 30)
    c = runner.compare([a.run_id, b.run_id], "accuracy")
    assert c["differing_hyperparameters"] == ["lr"] and c["best_run_id"] in (a.run_id, b.run_id)


# ---------------- MCP + CLI

def test_mcp_hpo_and_diagnose(toy):
    from mcp import Client
    from autoforge.serve.mcp_server import build_server

    plan = toy("score").model_dump(mode="json")

    async def go():
        async with Client(build_server("ai:test")) as c:
            call = lambda n, **a: c.call_tool(n, a)
            r = (await call("hpo_start", plan=plan, metric="score", n_trials=3)).structured_content
            assert r["ok"], r
            done = (await call("hpo_wait", study_id=r["data"]["study_id"], timeout_seconds=120)).structured_content
            assert done["data"]["status"] == "succeeded", done
            d = (await call("run_diagnose", run_id=done["data"]["best_run_id"])).structured_content
            assert d["ok"] and d["data"]["status"] == "succeeded"
            cmp = (await call("run_compare", run_ids=[t["run_id"] for t in done["data"]["trials"]],
                              metric="score")).structured_content
            assert cmp["data"]["best_run_id"] == done["data"]["best_run_id"]
            tools = {t.name for t in (await c.list_tools()).tools}
            assert {"hpo_start", "hpo_status", "hpo_stop", "run_adjust"} <= tools
    asyncio.run(go())


def test_cli_hpo_wait(toy, tmp_path):
    p = tmp_path / "plan.yaml"
    import yaml
    p.write_text(yaml.safe_dump(toy("score").model_dump(mode="json")))
    r = CliRunner().invoke(app, ["hpo", "start", str(p), "-m", "score", "-n", "3", "--wait"])
    assert r.exit_code == 0, r.output
    assert "trial 0 (baseline)" in r.output and "best params" in r.output


@pytest.mark.parametrize("metric,direction", [
    ("wer", "minimize"), ("val_cer", "minimize"), ("wape", "minimize"), ("mape", "minimize"), ("val_loss", "minimize"),
    ("map50", "maximize"), ("miou", "maximize"), ("recall_at_k", "maximize"), ("lower_bound", "maximize"),
    ("mean_reward", "maximize"), ("reward_accuracy", "maximize")])
def test_infer_direction(metric, direction):
    assert hpo.infer_direction(metric) == direction
