import pytest

from autoforge import api
from autoforge.data import registry
from autoforge.engine import runner
from autoforge.policy.privacy import PrivacyDenied, brain_from_actor
from autoforge.schemas.core import BrainKind, Sensitivity, TaskSpec, TrainingPlan


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "people" / "data.csv"
    p.parent.mkdir()
    rows = ["name,age,city,label"] + [f"person{i},{20 + i % 30},{'AB'[i % 2]},{i % 2}" for i in range(50)]
    p.write_text("\n".join(rows) + "\n")
    return p


def test_brain_from_actor():
    assert brain_from_actor("human") is BrainKind.human
    assert brain_from_actor("ai:local:qwen") is BrainKind.local
    assert brain_from_actor("ai:local") is BrainKind.local
    assert brain_from_actor("ai:claude") is BrainKind.api
    assert brain_from_actor("ai:localhost-hack") is BrainKind.api


def test_registry_default_and_directory_marks(csv_file):
    assert registry.sensitivity_of(csv_file) is Sensitivity.internal
    registry.mark(csv_file.parent, Sensitivity.sensitive)
    assert registry.sensitivity_of(csv_file) is Sensitivity.sensitive
    registry.mark(csv_file, Sensitivity.public)  # closest mark wins
    assert registry.sensitivity_of(csv_file) is Sensitivity.public


def test_profile_internal_has_stats_but_no_rows(csv_file):
    prof = api.data_profile(str(csv_file), BrainKind.subscription)
    assert prof["access"] == "summary" and prof["rows"] == 50
    assert "sample_rows" not in prof
    cols = prof["columns"]
    assert cols["age"]["dtype"] == "numeric" and cols["age"]["min"] == 20
    assert cols["city"]["top_values"] == {"A": 25, "B": 25}
    assert cols["name"]["distinct"] == ">20" and "top_values" not in cols["name"]  # identifiers hidden


def test_profile_public_and_human_get_rows(csv_file):
    assert "sample_rows" in api.data_profile(str(csv_file), BrainKind.human)
    registry.mark(csv_file, Sensitivity.public)
    assert len(api.data_profile(str(csv_file), BrainKind.subscription)["sample_rows"]) == 5


def test_sensitive_denied_for_cloud_allowed_for_local(csv_file):
    registry.mark(csv_file, Sensitivity.sensitive)
    with pytest.raises(PrivacyDenied):
        api.data_profile(str(csv_file), BrainKind.subscription)
    assert api.data_profile(str(csv_file), BrainKind.local)["access"] == "full"


def test_sensitive_data_cannot_be_trained_by_cloud_ai(csv_file):
    registry.mark(csv_file, Sensitivity.sensitive)
    plan = TrainingPlan(task=TaskSpec(task_type="toy", input_modality="x", output_modality="y", metric="m"),
                        data_path=str(csv_file), pack="whatever", recipe="r")
    with pytest.raises(PrivacyDenied):
        runner.create_run(plan, actor="ai:claude")
    with pytest.raises(PrivacyDenied):
        api.plan_validate(plan.model_dump(), "ai:mcp", BrainKind.subscription)
