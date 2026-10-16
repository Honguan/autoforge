"""Agent evaluation: the scorer, and a full run with a scripted brain (no model, nothing trained)."""

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autoforge.agents.providers import BrainSpec, Provider, ToolCall, Turn
from autoforge.cli.main import app
from autoforge.evals.agent_eval import Observed, run_eval, score
from autoforge.evals.cases import CASES, by_id

REPO = Path(__file__).resolve().parents[1]
CASE = {c.id: c for c in CASES}


def plan(task, pack, recipe, target=None):
    return {"task": {"task_type": task, "input_modality": "x", "output_modality": "y", "target": target},
            "data_path": "/d", "pack": pack, "recipe": recipe}


@pytest.mark.parametrize("case_id,obs,local,gpu,verdict", [
    ("churn", Observed(plan=plan("tabular-classification", "tabular-classic", "random_forest", "churned")), False, False, "pass"),
    ("churn", Observed(plan=plan("tabular-classification", "tabular-classic", "random_forest", "tenure_months")), False, False, "partial"),
    ("churn", Observed(plan=plan("tabular-regression", "tabular-classic", "random_forest", "churned")), False, False, "fail"),
    ("churn", Observed(validated_plan=plan("tabular-classification", "tabular-classic", "logistic_regression", "churned")), False, False, "partial"),
    ("churn", Observed(stop="max_steps"), False, False, "fail"),
    ("review-sentiment", Observed(plan=plan("text-classification", "text-classification", "transformer_finetune", "sentiment")), False, False, "partial"),
    ("labelled-fraud", Observed(plan=plan("anomaly-detection", "anomaly-detection", "isolation_forest")), False, False, "fail"),
    ("store-sales", Observed(plan=plan("timeseries-forecast", "timeseries-forecast", "lag_gbm", "revenue")), False, False, "partial"),
    ("support-bot", Observed(final_text="This needs an NVIDIA GPU with 8 GB VRAM."), False, False, "pass"),
    ("support-bot", Observed(final_text="I could not decide."), False, False, "fail"),
    ("support-bot", Observed(final_text="This needs a GPU."), False, True, "fail"),  # a GPU exists: should have planned
    ("sensitive-patients", Observed(privacy_denied=True), False, False, "pass"),
    ("sensitive-patients", Observed(plan=plan("tabular-classification", "tabular-classic", "random_forest", "readmitted")), False, False, "fail"),
    ("sensitive-patients", Observed(), False, False, "partial"),
    ("sensitive-patients", Observed(plan=plan("tabular-classification", "tabular-classic", "random_forest", "readmitted")), True, False, "pass"),
    ("churn", Observed(error="ProviderError: boom"), False, False, "error"),
])
def test_score(case_id, obs, local, gpu, verdict):
    assert score(CASE[case_id], obs, local, gpu)[0] == verdict


def test_every_case_writes_its_data(tmp_path):
    for c in CASES:
        p = c.make_data(tmp_path / c.id)
        assert p.exists(), c.id
        if p.suffix == ".csv" and c.target:
            assert c.target in p.read_text(encoding="utf-8").splitlines()[0].split(","), c.id
    png = next((tmp_path / "defect-photos").rglob("*.png")).read_bytes()
    assert png.startswith(b"\x89PNG") and b"IEND" in png
    with pytest.raises(ValueError, match="unknown case"):
        by_id(["nope"])


class Oracle(Provider):
    """Profiles the data, installs the right local pack, proposes the expected plan (or stops on privacy)."""

    PLANS = {"churn": ("tabular-classification", "tabular-classic", "random_forest", "churned"),
             "sensitive-patients": ("tabular-classification", "tabular-classic", "random_forest", "readmitted")}

    def start(self, system, tools):
        self.step, self.results = 0, []

    def add_user(self, text):
        self.data = re.search(r"Data: (.+)$", text).group(1)
        self.case = Path(self.data).parent.name

    def add_tool_results(self, results):
        self.results.append(json.loads(results[0].content))

    async def complete(self):
        self.step += 1
        if self.step == 1:
            return Turn(tool_calls=[ToolCall("1", "data_profile", {"path": self.data})], stop="tool_use")
        if not self.results[-1]["ok"]:
            return Turn(text=f"Stopped: {self.results[-1]['error']['code']}", stop="end")
        if self.step == 2:
            return Turn(tool_calls=[ToolCall("2", "pack_install", {"source": str(REPO / "packs" / "tabular-classic")})],
                        stop="tool_use")
        task, pack, recipe, target = self.PLANS[self.case]
        p = plan(task, pack, recipe, target) | {"data_path": self.data}
        return Turn(tool_calls=[ToolCall("3", "run_start", {"plan": p})], stop="tool_use")


def test_run_eval_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path / "real-home"))
    spec = BrainSpec("anthropic", "oracle")
    seen = []
    report = run_eval(by_id(["churn", "sensitive-patients"]), spec, lambda s: Oracle(), on_progress=seen.append)
    assert report["counts"] == {"pass": 2, "partial": 0, "fail": 0, "error": 0} and report["met_target"]
    churn, sens = report["cases"]
    assert (churn["pack"], churn["recipe"], churn["target"]) == ("tabular-classic", "random_forest", "churned")
    assert churn["tools"] == ["data_profile", "pack_install", "run_start"]  # stopped right after the gated call
    assert sens["tools"] == ["data_profile"] and "PRIVACY_DENIED" in sens["reason"]
    out = Path(report["report_dir"])
    assert out.parent.parent == tmp_path / "real-home" and (out / "report.md").is_file()
    assert Path(churn["transcript"]).parent == out and Path(churn["transcript"]).is_file()
    assert not (tmp_path / "real-home" / "runs").exists()  # nothing trained, nothing left in the real home
    assert [r.id for r in seen] == ["churn", "sensitive-patients"]


def test_cli_lists_cases_and_needs_a_working_brain(monkeypatch):
    r = CliRunner().invoke(app, ["eval", "agent", "--list", "--json"])
    assert r.exit_code == 0 and len(json.loads(r.stdout)["data"]) == len(CASES)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = CliRunner().invoke(app, ["eval", "agent", "--brain", "openai:gpt-x", "--json"])
    assert r.exit_code != 0 and json.loads(r.stdout)["error"]["code"] == "PROVIDER_ERROR"


def test_profile_shows_folder_files_and_jsonl_fields(tmp_path):
    from autoforge.data.profile import profile
    from autoforge.policy.privacy import Access

    d = CASE["shelf-boxes"].make_data(tmp_path / "a")
    assert "annotations.csv" in profile(d, Access.summary)["top_level_files"]
    j = profile(CASE["support-bot"].make_data(tmp_path / "b"), Access.summary)
    assert j["format"] == "jsonl" and j["records"] == 200 and set(j["fields"]) == {"prompt", "response"}
    assert "訂單" not in json.dumps(j, ensure_ascii=False)  # field names only, never values


def test_gpu_check_harness_on_cpu(tmp_path, monkeypatch):
    """The real checks need an NVIDIA GPU; on CPU we run the same harness with tiny settings."""
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("PIL")
    from autoforge.evals.gpu_check import by_id as gpu_by_id, run_checks

    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path / "real-home"))
    tiny = {"image-classification": {"pretrained": False, "image_size": 32, "epochs": 1, "batch_size": 16}}
    seen = []
    report = run_checks(gpu_by_id(["image-classification", "llm-qlora"]), device="cpu", packs_dir=REPO / "packs",
                        timeout=300, hp_override=tiny, on_progress=seen.append)
    img, llm = report["checks"]
    assert img["device"] == "cpu" and img["run_id"] and "accuracy" in img["metrics"], img
    assert img["verdict"] in ("pass", "fail") and json.loads(img["prediction"])["prediction"] in ("red", "green", "blue")
    # sft_lora needs CUDA: without a GPU the plan is refused (or bitsandbytes is missing) — reported, not crashed
    assert llm["verdict"] in ("skip", "error") and llm["reason"]
    out = Path(report["report_dir"])
    assert out.parent.parent == tmp_path / "real-home" and "| image-classification |" in (out / "report.md").read_text()
    assert [r.id for r in seen] == ["image-classification", "llm-qlora"]


def test_gpu_check_data_generators(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("soundfile")
    from autoforge.evals.gpu_check import CHECKS

    for c in CHECKS:
        p = c.make_data(tmp_path / c.id)
        assert p.exists(), c.id
        row = c.predict_row(tmp_path / c.id)
        for v in row.values():
            if isinstance(v, str) and v.startswith(str(tmp_path)):
                assert Path(v).exists(), (c.id, v)


def test_eval_gpu_refuses_without_gpu():
    from autoforge.hardware.detect import detect_hardware

    if detect_hardware().best_device == "cuda":
        pytest.skip("this machine has a GPU")
    r = CliRunner().invoke(app, ["eval", "gpu", "--json"])
    assert r.exit_code != 0 and json.loads(r.stdout)["error"]["code"] == "NO_GPU"
    r = CliRunner().invoke(app, ["eval", "gpu", "--list", "--json"])
    assert r.exit_code == 0 and len(json.loads(r.stdout)["data"]) == 10
