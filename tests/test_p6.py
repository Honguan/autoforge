import asyncio
import os
import sys
import zipfile
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters
from starlette.testclient import TestClient

from autoforge import api
from autoforge.data import registry as data_registry
from autoforge.engine import runner
from autoforge.engine.validate import load_plan_file
from autoforge.packs import installer
from autoforge.registry import models as reg
from autoforge.schemas.core import Sensitivity, TaskSpec, TrainingPlan
from autoforge.serve.model_mcp import build_model_server
from autoforge.serve.mcp_server import build_server
from autoforge.serve.rest import create_app

pytest.importorskip("sklearn")
REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "engine-pack"
HIGH_RISK = {"tenure_months": 2, "monthly_fee": 110.0, "contract": "month-to-month", "support_calls": 5}
LOYAL = {"tenure_months": 70, "monthly_fee": 20.0, "contract": "two-year", "support_calls": 0}


def train(recipe="logistic_regression", data=None):
    plan = load_plan_file(REPO / "examples" / "churn_plan.yaml")
    plan = plan.model_copy(update={"recipe": recipe, "hyperparameters": {}} | ({"data_path": str(data)} if data else {}))
    s = runner.create_run(plan)
    runner.start(s)
    v = runner.wait(s, 120)
    assert v.status.value == "succeeded", s.logs_tail(40)
    return s.run_id


@pytest.fixture
def churn():
    installer.install(str(REPO / "packs" / "tabular-classic"))
    return train()


def test_register_card_and_versions(churn):
    card = reg.register(churn, "churn", actor="ai:test", description="Churn risk")
    assert card.ref == "churn:v1" and card.servable and card.labels == ["0", "1"]
    assert [f.name for f in card.features] == ["tenure_months", "monthly_fee", "contract", "support_calls"]
    contract = next(f for f in card.features if f.name == "contract")
    assert contract.dtype == "categorical" and set(contract.known_values) == {"month-to-month", "one-year", "two-year"}
    assert card.metrics["f1_macro"] > 0.5 and card.pack_version == "0.3.0"
    assert reg.register(churn, "churn").version == 2
    assert reg.get_card("churn").version == 2 and reg.get_card("churn:v1").version == 1
    with pytest.raises(reg.ModelNotFound):
        reg.get_card("churn:v9")
    with pytest.raises(reg.ModelError):
        reg.register(churn, "Bad Name")


def test_predict_and_validation(churn):
    reg.register(churn, "churn")
    out = reg.predict("churn", [HIGH_RISK, LOYAL])
    p = out["predictions"]
    assert p[0]["prediction"] == 1 and p[1]["prediction"] == 0
    assert abs(sum(p[0]["probabilities"].values()) - 1) < 1e-6
    w = reg.predict("churn", [HIGH_RISK | {"contract": "weekly", "extra": 1}])["warnings"]
    assert any("not seen in training" in x for x in w) and any("unknown fields" in x for x in w)
    with pytest.raises(reg.InvalidInput) as e:
        reg.predict("churn", [{"tenure_months": 1}])
    assert "missing" in str(e.value) and e.value.details["features"]
    with pytest.raises(reg.InvalidInput):
        reg.predict("churn", [HIGH_RISK | {"monthly_fee": "lots"}])
    with pytest.raises(reg.InvalidInput):
        reg.predict("churn", [])


def test_registered_model_uses_its_own_code_snapshot(churn):
    reg.register(churn, "churn")
    rec, pack = installer.get_installed("tabular-classic")
    trainer = Path(rec.path) / "trainers" / "sklearn_trainers.py"
    trainer.write_text(trainer.read_text().replace("def predict(", "def predict_broken("))  # break the installed pack
    reg._CACHE.clear()
    assert reg.predict("churn", [LOYAL])["predictions"][0]["prediction"] == 0


def test_not_servable_and_failed_runs(tmp_path):
    installer.install(str(FIXTURE))
    data = tmp_path / "d.txt"
    data.write_text("x")
    plan = TrainingPlan(task=TaskSpec(task_type="toy", input_modality="x", output_modality="y", metric="m"),
                        data_path=str(data), pack="engine-test", recipe="quick")
    s = runner.create_run(plan)
    runner.start(s)
    runner.wait(s, 30)
    card = reg.register(s.run_id, "toy")
    assert card.servable is False
    with pytest.raises(reg.ModelNotServable):
        reg.predict("toy", [{"a": 1}])
    s2 = runner.create_run(plan.model_copy(update={"recipe": "crash"}))
    runner.start(s2)
    runner.wait(s2, 30)
    with pytest.raises(reg.ModelError) as e:
        reg.register(s2.run_id, "toy")
    assert "only succeeded runs" in str(e.value)


def test_sensitive_training_data_not_copied_into_card(churn, tmp_path):
    data = tmp_path / "secret.csv"
    data.write_text((REPO / "examples" / "data" / "churn.csv").read_text())
    data_registry.mark(data, Sensitivity.sensitive)
    card = reg.register(train(data=data), "secret")
    assert card.data_sensitivity is Sensitivity.sensitive and card.labels is None
    assert all(f.known_values is None for f in card.features)


def test_bundle_export_import_roundtrip(churn, tmp_path):
    reg.register(churn, "churn")
    res = reg.export("churn", "bundle", tmp_path / "churn.zip")
    assert len(res["sha256"]) == 64
    card = reg.import_bundle(tmp_path / "churn.zip", name="churn-copy")
    assert card.ref == "churn-copy:v1"
    a = reg.predict("churn", [HIGH_RISK])["predictions"]
    assert reg.predict("churn-copy", [HIGH_RISK])["predictions"] == a
    bad = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../escape.txt", "x")
        z.writestr("card.json", (reg._version_dir("churn", 1) / "card.json").read_text())
    with pytest.raises(reg.ModelError):
        reg.import_bundle(bad)


def test_onnx_export_matches_predictions(churn, tmp_path):
    pytest.importorskip("skl2onnx")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    reg.register(churn, "churn")
    res = reg.export("churn", "onnx", tmp_path / "churn.onnx")
    sess = ort.InferenceSession(res["path"])
    rows = [HIGH_RISK, LOYAL]
    card = reg.get_card("churn")
    feeds = {f.name: np.array([[r[f.name]] for r in rows], dtype=np.float32 if f.dtype == "numeric" else object)
             for f in card.features}
    feeds = {k: (v.astype(str) if v.dtype == object else v) for k, v in feeds.items()}
    labels = sess.run(None, feeds)[0].ravel().tolist()
    assert labels == [p["prediction"] for p in reg.predict("churn", rows)["predictions"]]
    with pytest.raises(reg.ModelError):
        reg.export("churn", "gguf", tmp_path / "x")


def test_rest_api(churn):
    reg.register(churn, "churn")
    c = TestClient(create_app())
    assert c.get("/health").json()["ok"]
    assert [m["name"] for m in c.get("/v1/models").json()["data"]] == ["churn"]
    r = c.post("/v1/models/churn/predict", json={"rows": [HIGH_RISK]})
    assert r.status_code == 200 and r.json()["data"]["predictions"][0]["prediction"] == 1
    r = c.post("/v1/models/churn/predict", json={"rows": [{"tenure_months": 1}]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_INPUT"
    assert c.get("/v1/models/nope").status_code == 404
    locked = TestClient(create_app(["churn"], api_key="s3cret"))
    assert locked.get("/v1/models").status_code == 401
    ok = locked.post("/v1/models/churn/predict", json={"rows": [LOYAL]}, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert TestClient(create_app(["other"])).post("/v1/models/churn/predict", json={"rows": [LOYAL]}).status_code == 404


def test_main_mcp_register_and_predict(churn):
    async def go():
        async with Client(build_server("ai:test")) as c:
            call = lambda n, **a: c.call_tool(n, a)
            r = (await call("model_register", run_id=churn, name="churn")).structured_content
            assert r["ok"] and r["data"]["servable"], r
            p = (await call("model_predict", model="churn", rows=[HIGH_RISK])).structured_content
            assert p["data"]["predictions"][0]["prediction"] == 1
            info = (await call("model_info", model="churn")).structured_content
            assert "onnx" in info["data"]["export_formats"]
    asyncio.run(go())


def test_models_mcp_server_over_stdio(churn, autoforge_home):
    """P6 acceptance: another AI calls the trained model over MCP, like Claude Desktop would."""
    reg.register(churn, "churn", description="Probability that a customer cancels.")
    params = StdioServerParameters(command=sys.executable, args=["-m", "autoforge.cli.main", "serve-mcp"],
                                   env={**os.environ, "AUTOFORGE_HOME": str(autoforge_home)})

    async def go():
        async with Client(params) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            assert set(tools) == {"predict_churn", "list_models"}
            t = tools["predict_churn"]
            assert "Probability that a customer cancels" in t.description and "0, 1" in t.description
            assert t.annotations.read_only_hint
            r = await c.call_tool("predict_churn", {"rows": [HIGH_RISK, LOYAL]})
            return r.structured_content
    out = asyncio.run(go())
    assert [p["prediction"] for p in out["data"]["predictions"]] == [1, 0]
