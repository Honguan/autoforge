import asyncio
import json
import math
from pathlib import Path

import pytest
from mcp import Client
from typer.testing import CliRunner

from autoforge.cli.main import app
from autoforge.engine import runner
from autoforge.packs import installer
from autoforge.packs.authoring import AuthoringError, create_from_files, scaffold
from autoforge.registry import models as reg
from autoforge.schemas.core import TaskSpec, TrainingPlan
from autoforge.serve.model_mcp import build_model_server
from autoforge.serve.mcp_server import build_server

REPO = Path(__file__).resolve().parents[1]
PACKS = ["tabular-classic", "text-classification", "llm-finetune", "image-classification",
         "timeseries-forecast", "anomaly-detection"]


@pytest.mark.parametrize("name", PACKS)
def test_every_bundled_pack_is_valid_and_documented(name):
    from autoforge.packs.manifest import load_pack

    pack = load_pack(REPO / "packs" / name)
    k = pack.knowledge()
    for section in ("When to use", "When NOT to use", "Common failures"):
        assert section in k, f"{name}: KNOWLEDGE.md lacks '{section}'"
    assert (REPO / "packs" / name / "tests").is_dir()
    assert all(r.predictor for r in pack.recipes), f"{name}: every recipe should be servable"


def test_scaffold_is_valid_and_testable(tmp_path):
    res = scaffold("my-pack", "my-task", tmp_path)
    assert res["valid"], res["problems"]
    r = CliRunner().invoke(app, ["pack", "test", res["path"], "--json"])
    assert r.exit_code == 0 and json.loads(r.stdout)["data"]["passed"] == 1
    with pytest.raises(AuthoringError):
        scaffold("my-pack", "my-task", tmp_path)


def test_pack_create_validates_and_rejects_unsafe_paths(tmp_path):
    src = Path(scaffold("seed", "demo", tmp_path)["path"])
    files = {str(p.relative_to(src)): p.read_text() for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    files["pack.yaml"] = files["pack.yaml"].replace("name: seed", "name: from-ai")
    res = create_from_files("from-ai", files)
    assert res["valid"], res["problems"]
    bad = dict(files)
    del bad["KNOWLEDGE.md"]
    assert not create_from_files("from-ai", bad, overwrite=True)["valid"]
    with pytest.raises(AuthoringError):
        create_from_files("from-ai", files)  # exists, no overwrite
    with pytest.raises(AuthoringError):
        create_from_files("evil", {"../escape.py": "x"})


def test_pack_create_over_mcp(tmp_path):
    src = Path(scaffold("seed", "demo", tmp_path)["path"])
    files = {str(p.relative_to(src)): p.read_text() for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    files["pack.yaml"] = files["pack.yaml"].replace("name: seed", "name: ai-pack")

    async def go():
        async with Client(build_server("ai:test")) as c:
            created = (await c.call_tool("pack_create", {"name": "ai-pack", "files": files})).structured_content
            assert created["ok"] and created["data"]["valid"], created
            tested = (await c.call_tool("pack_test", {"target": created["data"]["path"]})).structured_content
            assert tested["data"]["ok"], tested
            inst = (await c.call_tool("pack_install", {"source": created["data"]["path"]})).structured_content
            assert inst["ok"] and inst["data"]["name"] == "ai-pack"
    asyncio.run(go())


def _run(plan):
    s = runner.create_run(plan)
    runner.start(s)
    v = runner.wait(s, 180)
    assert v.status.value == "succeeded", s.logs_tail(40)
    return s.run_id, v


def test_text_model_end_to_end_with_io_json(tmp_path):
    pytest.importorskip("sklearn")
    installer.install(str(REPO / "packs" / "text-classification"))
    data = tmp_path / "t.csv"
    rows = ["text,label"] + [f"great happy love {i},pos" if i % 2 else f"awful angry broken {i},neg" for i in range(60)]
    data.write_text("\n".join(rows))
    plan = TrainingPlan(task=TaskSpec(task_type="text-classification", input_modality="text", output_modality="label",
                                      target="label", metric="f1_macro"), data_path=str(data),
                        pack="text-classification", recipe="tfidf_linear")
    run_id, v = _run(plan)
    assert v.metrics["f1_macro"] > 0.9
    card = reg.register(run_id, "sentiment")
    assert [(f.name, f.dtype) for f in card.features] == [("text", "text")] and card.labels == ["neg", "pos"]

    async def go():
        async with Client(build_model_server()) as c:
            tool = next(t for t in (await c.list_tools()).tools if t.name == "predict_sentiment")
            row = tool.input_schema["$defs"]["predict_sentiment_row"]["properties"]["text"]
            assert row["type"] == "string"
            r = await c.call_tool("predict_sentiment", {"rows": [{"text": "love it, great"}]})
            return r.structured_content
    assert asyncio.run(go())["data"]["predictions"][0]["prediction"] == "pos"


def test_forecast_model_end_to_end(tmp_path):
    pd = pytest.importorskip("pandas")
    installer.install(str(REPO / "packs" / "timeseries-forecast"))
    data = tmp_path / "s.csv"
    pd.DataFrame({"date": pd.date_range("2025-01-01", periods=200, freq="D"),
                  "load": [50 + 10 * math.sin(2 * math.pi * i / 7) for i in range(200)]}).to_csv(data, index=False)
    plan = TrainingPlan(task=TaskSpec(task_type="timeseries-forecast", input_modality="series", output_modality="values",
                                      target="load", metric="mae"), data_path=str(data),
                        pack="timeseries-forecast", recipe="lag_gbm")
    run_id, v = _run(plan)
    assert v.metrics["skill_vs_naive"] > 0
    reg.register(run_id, "load")
    out = reg.predict("load", [{"horizon": 3}])
    assert len(out["predictions"][0]["prediction"]) == 3


def test_directory_export_is_zipped(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    installer.install(str(REPO / "packs" / "anomaly-detection"))
    pd = pytest.importorskip("pandas")
    data = tmp_path / "a.csv"
    pd.DataFrame({"x": list(range(100)), "y": [1.0] * 100}).to_csv(data, index=False)
    plan = TrainingPlan(task=TaskSpec(task_type="anomaly-detection", input_modality="tabular", output_modality="flag",
                                      metric="anomaly_rate"), data_path=str(data),
                        pack="anomaly-detection", recipe="isolation_forest")
    run_id, _ = _run(plan)
    card = reg.register(run_id, "anom")
    vdir = reg._version_dir("anom", card.version)
    (vdir / "code" / "direxp.py").write_text(
        "from pathlib import Path\ndef export(model_dir, out, features):\n"
        "    Path(out).mkdir(); (Path(out) / 'w.bin').write_bytes(b'x' * 10)\n")
    data_card = json.loads((vdir / "card.json").read_text())
    data_card["exporters"]["dir"] = "code/direxp.py:export"
    (vdir / "card.json").write_text(json.dumps(data_card))
    res = reg.export("anom", "dir", tmp_path / "out")
    assert res["path"].endswith(".zip") and Path(res["path"]).is_file()


@pytest.mark.parametrize("name", PACKS)
def test_pack_smoke_tests_pass(name):
    """Each pack's own tests/, run in a separate process exactly like `autoforge pack test`."""
    from autoforge.packs.authoring import test_pack

    res = test_pack(REPO / "packs" / name, timeout=600)
    assert res["ok"], "\n".join(res["output_tail"])
    assert res["passed"] + res["skipped"] >= 1  # deep-learning packs skip when torch is not installed
