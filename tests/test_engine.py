import time
from pathlib import Path

import pytest

from autoforge.engine import runner
from autoforge.engine.store import RunStore, wait_for
from autoforge.engine.validate import PlanInvalid, check_plan, load_plan_file
from autoforge.packs import installer
from autoforge.schemas.core import RunStatus, TaskSpec, TrainingPlan

FIXTURE = Path(__file__).parent / "fixtures" / "engine-pack"


@pytest.fixture
def toy(tmp_path):
    installer.install(str(FIXTURE))
    data = tmp_path / "data.txt"
    data.write_text("x")

    def make(recipe="quick", **kw):
        return TrainingPlan(task=TaskSpec(task_type="toy", input_modality="x", output_modality="y", metric="loss"),
                            data_path=str(data), pack="engine-test", recipe=recipe, **kw)
    return make


def run(plan, actor="human"):
    store = runner.create_run(plan, actor)
    runner.start(store, actor)
    return store


def test_quick_run_succeeds_with_merged_hyperparameters(toy):
    store = run(toy(hyperparameters={"lr": 0.5}))
    view = runner.wait(store, 30)
    assert view.status is RunStatus.succeeded, store.logs_tail(30)
    assert view.metrics == {"accuracy": 0.9}
    assert Path(view.model_path).read_text() == "lr=0.5 device=cpu"
    rec = store.record()
    assert rec.plan.hyperparameters == {"steps": 3, "lr": 0.5}
    assert rec.git_tracked and (store.dir / ".git").is_dir()
    assert [a["action"] for a in store.audit_log()] == ["create", "start"]


def test_crash_is_reported(toy):
    view = runner.wait(run(toy("crash")), 30)
    assert view.status is RunStatus.failed and "boom" in view.error


def test_metrics_summary(toy):
    store = run(toy("slow", hyperparameters={"steps": 4}))
    assert runner.wait(store, 30).status is RunStatus.succeeded
    s = runner.metrics_summary(store)
    assert s["loss"]["min"] == 0.25 and s["loss"]["count"] == 5  # 4 steps + final metric


def test_pause_resume_stop(toy):
    store = run(toy("slow", hyperparameters={"steps": 100000}), actor="ai:test")
    assert wait_for(lambda: len(store.events()) > 3, 30)
    assert runner.pause(store, "human").status is RunStatus.paused
    time.sleep(0.3)
    n = len(store.events())
    time.sleep(0.5)
    assert len(store.events()) == n, "no progress while paused"
    assert runner.resume(store, "ai:test").status is RunStatus.running
    assert wait_for(lambda: len(store.events()) > n, 10)
    view = runner.stop(store, "human", grace_seconds=10)
    assert view.status is RunStatus.stopped
    actors = [(a["actor"], a["action"]) for a in store.audit_log()]
    assert actors == [("ai:test", "create"), ("ai:test", "start"), ("human", "pause"),
                      ("ai:test", "resume"), ("human", "stop")]


def test_stop_terminates_non_cooperative_trainer(toy):
    store = run(toy("stubborn"))
    assert wait_for(lambda: store.state().status is RunStatus.running, 30)
    view = runner.stop(store, grace_seconds=0.5)
    assert view.status is RunStatus.stopped
    assert any(a["action"] == "terminate" for a in store.audit_log())


def test_control_on_finished_run_is_rejected(toy):
    store = run(toy())
    runner.wait(store, 30)
    with pytest.raises(runner.RunStateError):
        runner.pause(store)


def test_retry_resumes_from_checkpoint(toy):
    store = run(toy("slow", hyperparameters={"steps": 100000}))
    assert wait_for(lambda: len(list((store.dir / "checkpoints").glob("*"))) >= 3, 30)
    runner.stop(store)
    new = runner.retry(store)
    rec = new.record()
    assert rec.retry_of == store.run_id and rec.resume_from
    # Make the retried run short so it finishes: shrink steps via a fresh plan file? Instead stop it once resumed.
    runner.start(new)
    assert wait_for(lambda: any("resuming from step" in e.get("message", "") for e in new.events()), 30)
    runner.stop(new)


def test_custom_code_is_snapshotted_and_versioned(toy, tmp_path):
    code = tmp_path / "my_trainer.py"
    code.write_text("def train(ctx):\n    return {'metrics': {'score': 1.0}}\n")
    plan = toy()
    plan = plan.model_copy(update={"pack": None, "recipe": None, "custom_code_path": str(code)})
    store = run(plan, actor="ai:claude")
    code.write_text("def train(ctx):\n    raise SystemExit('changed after start')\n")  # must not affect the run
    view = runner.wait(store, 30)
    assert view.status is RunStatus.succeeded and view.metrics == {"score": 1.0}
    assert (store.dir / "code" / "my_trainer.py").is_file()


def test_plan_validation_problems(toy):
    c = check_plan(toy(recipe="nope"))
    assert not c.ok and any("recipe 'nope'" in p for p in c.problems)
    c = check_plan(toy().model_copy(update={"data_path": "/no/such/file"}))
    assert any("data_path not found" in p for p in c.problems)
    c = check_plan(toy(device="cuda"))
    assert any("CPU-only" in p for p in c.problems)
    c = check_plan(toy(hyperparameters={"wat": 1}))
    assert c.ok and any("wat" in w for w in c.warnings)
    with pytest.raises(PlanInvalid):
        runner.create_run(toy(recipe="nope"))


def test_load_plan_file_resolves_relative_paths(tmp_path):
    (tmp_path / "d.csv").write_text("a,b\n1,2\n")
    (tmp_path / "plan.yaml").write_text(
        "task: {task_type: toy, input_modality: x, output_modality: y, metric: m, target: b}\n"
        "data_path: d.csv\npack: engine-test\nrecipe: quick\n")
    plan = load_plan_file(tmp_path / "plan.yaml")
    assert plan.data_path == str((tmp_path / "d.csv").resolve())
    with pytest.raises(PlanInvalid):
        (tmp_path / "bad.yaml").write_text("task: 1\n")
        load_plan_file(tmp_path / "bad.yaml")


def test_unknown_run_id():
    with pytest.raises(Exception) as e:
        RunStore.open("nope")
    assert e.value.code == "RUN_NOT_FOUND"


def test_git_commit_ignores_temp_files_written_by_the_worker(toy):
    """Regression: `git add -A` raced the worker's atomic temp files and silently skipped commits."""
    import subprocess

    store = runner.create_run(toy())
    (store.dir / "state.json.tmp12345").write_text("{}")  # what an atomic write leaves for an instant
    (store.dir / "adjustments.yaml").write_text("lr: 0.5\n")
    assert store.git_commit("adjust {'lr': 0.5}", "human")
    tracked = subprocess.run(["git", "ls-files"], cwd=store.dir, capture_output=True, text=True).stdout.split()
    assert "adjustments.yaml" in tracked and "plan.yaml" in tracked
    assert not any(".tmp" in t for t in tracked)
