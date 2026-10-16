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
