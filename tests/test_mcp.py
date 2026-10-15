import asyncio
import os
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from autoforge.data import registry
from autoforge.schemas.core import Sensitivity
from autoforge.serve.mcp_server import build_server

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "engine-pack"


def session(actor="ai:test"):
    """Run a coroutine `fn(call)` against an in-process server; call(tool, **args) -> envelope."""
    def runner(fn):
        async def go():
            async with Client(build_server(actor)) as client:
                async def call(_tool, **args):
                    res = await client.call_tool(_tool, args)
                    return res.structured_content
                return await fn(call)
        return asyncio.run(go())
    return runner


def toy_plan(data, **kw):
    return {"task": {"task_type": "toy", "input_modality": "x", "output_modality": "y", "metric": "loss"},
            "data_path": str(data), "pack": "engine-test", "recipe": "quick"} | kw


def test_tools_are_listed_with_annotations():
    async def go():
        async with Client(build_server()) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            return tools
    tools = asyncio.run(go())
    for name in ["hardware_profile", "pack_search", "pack_install", "pack_info", "data_profile",
                 "trainer_write", "plan_validate", "run_start", "run_wait", "run_stop", "run_retry"]:
        assert name in tools
    assert tools["run_stop"].annotations.destructive_hint is True
    assert tools["hardware_profile"].annotations.read_only_hint is True


def test_full_training_workflow_over_mcp(tmp_path):
    data = tmp_path / "d.txt"
    data.write_text("x")

    async def flow(call):
        hw = await call("hardware_profile")
        assert hw["ok"] and hw["data"]["best_device"] in ("cpu", "cuda")
        assert (await call("pack_install", source=str(FIXTURE)))["ok"]
        info = await call("pack_info", name="engine-test")
        assert "stubborn" in info["data"]["knowledge"]
        check = await call("plan_validate", plan=toy_plan(data))
        assert check["ok"] and check["data"]["ok"], check
        started = await call("run_start", plan=toy_plan(data, hyperparameters={"lr": 0.2}))
        assert started["ok"], started
        run_id = started["data"]["run_id"]
        done = await call("run_wait", run_id=run_id, timeout_seconds=60)
        assert done["data"]["status"] == "succeeded", done
        audit = await call("run_audit", run_id=run_id)
        assert {a["actor"] for a in audit["data"]} == {"ai:test"}
        return done

    session()(flow)


def test_ai_written_trainer(tmp_path):
    data = tmp_path / "d.txt"
    data.write_text("x")
    code = "def train(ctx):\n    ctx.log_metric('score', 0.75)\n    return {'metrics': {'score': 0.75}}\n"

    async def flow(call):
        bad = await call("trainer_write", filename="x.py", source="def (:")
        assert bad["error"]["code"] == "BAD_REQUEST"
        w = await call("trainer_write", filename="my_trainer.py", source=code)
        assert w["ok"] and w["data"]["functions"] == ["train"]
        plan = toy_plan(data, pack=None, recipe=None, custom_code_path=w["data"]["path"])
        started = await call("run_start", plan=plan)
        done = await call("run_wait", run_id=started["data"]["run_id"], timeout_seconds=60)
        assert done["data"]["metrics"] == {"score": 0.75}

    session()(flow)


def test_errors_are_envelopes(tmp_path):
    async def flow(call):
        r = await call("run_status", run_id="nope")
        assert r["ok"] is False and r["error"]["code"] == "RUN_NOT_FOUND"
        r = await call("plan_validate", plan={"task": 1})
        assert r["error"]["code"] == "PLAN_INVALID" and r["data"]["problems"]
        r = await call("schema_get", name="TrainingPlan")
        assert "data_path" in r["data"]["properties"]

    session()(flow)


def test_privacy_enforced_over_mcp(tmp_path):
    data = tmp_path / "secret.csv"
    data.write_text("a,b\n1,2\n")
    registry.mark(data, Sensitivity.sensitive)

    async def flow(call):
        for r in [await call("data_profile", path=str(data)),
                  await call("plan_validate", plan=toy_plan(data)),
                  await call("run_start", plan=toy_plan(data))]:
            assert r["ok"] is False and r["error"]["code"] == "PRIVACY_DENIED", r

    session("ai:claude")(flow)

    async def local_flow(call):
        r = await call("data_profile", path=str(data))
        assert r["ok"] and r["data"]["access"] == "full"

    session("ai:local:qwen")(local_flow)


def test_stdio_server_end_to_end(autoforge_home):
    """Launch `autoforge mcp` exactly like Claude Desktop/Code would."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "autoforge.cli.main", "mcp", "--actor", "ai:stdio-test"],
        env={**os.environ, "AUTOFORGE_HOME": str(autoforge_home)},
    )

    async def go():
        async with Client(params) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            res = await client.call_tool("pack_list", {})
            return names, res.structured_content
    names, listed = asyncio.run(go())
    assert "run_start" in names and listed == {"ok": True, "data": [], "error": None}
