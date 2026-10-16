"""GUI agent chat: multi-turn conversations, approval in the page, stop, privacy as the brain's actor."""

import json
import time

from starlette.testclient import TestClient

from autoforge.agents.providers import Provider, ToolCall, Turn
from autoforge.data import registry
from autoforge.gui.app import create_app
from autoforge.schemas.core import Sensitivity

TOKEN = "t"


class Scripted(Provider):
    def __init__(self, script):
        self.script, self.user, self.results = list(script), [], []

    def start(self, system, tools):
        self.tools = [t.name for t in tools]

    def add_user(self, text):
        self.user.append(text)

    async def complete(self):
        return self.script.pop(0)(self.results[-1] if self.results else [])

    def add_tool_results(self, results):
        self.results.append(results)


def call(name, **args):
    return lambda _: Turn(tool_calls=[ToolCall(f"c-{name}", name, args)], stop="tool_use")


def say(text):
    return lambda _: Turn(text=text, stop="end")


def make(provider):
    c = TestClient(create_app(TOKEN, provider_factory=lambda spec: provider), base_url="http://127.0.0.1")
    c.headers["X-AutoForge-Token"] = TOKEN
    return c


def wait(c, chat_id, until, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        v = c.get(f"/api/agent/chats/{chat_id}").json()["data"]
        if until(v):
            return v
        time.sleep(0.05)
    raise AssertionError(f"timed out; last view: {json.dumps(v)[:800]}")


def kinds(view):
    return [e["kind"] for e in view["events"]]


def test_conversation_with_approval_and_follow_up(tmp_path):
    plan = {"task": {"task_type": "toy", "input_modality": "x", "output_modality": "y"},
            "data_path": str(tmp_path), "pack": "none", "recipe": "none"}
    p = Scripted([call("hardware_profile"), call("run_start", plan=plan),
                  say("OK, I will use a smaller model."), say("Second answer.")])
    with make(p) as c:
        chat = c.post("/api/agent/chats", json={"brain": "anthropic:test-model"}).json()["data"]
        assert chat["actor"] == "ai:anthropic:test-model" and chat["local"] is False
        cid = chat["id"]
        wait(c, cid, lambda v: v["status"] == "idle")

        r = c.post(f"/api/agent/chats/{cid}/messages", json={"text": "train a churn model", "data_path": str(tmp_path)})
        assert r.json()["ok"]
        busy = c.post(f"/api/agent/chats/{cid}/messages", json={"text": "again"})
        assert busy.status_code == 409 and busy.json()["error"]["code"] == "CHAT_ERROR"

        v = wait(c, cid, lambda v: v["status"] == "approval")
        assert v["pending"]["tool"] == "run_start" and "task:" in v["pending"]["summary"]
        c.post(f"/api/agent/chats/{cid}/approval", json={"approved": False, "feedback": "use less memory"})
        v = wait(c, cid, lambda v: v["status"] == "idle" and "done" in kinds(v))
        declined = json.loads(p.results[-1][0].content)
        assert declined["error"]["code"] == "USER_DECLINED" and "use less memory" in declined["error"]["message"]
        assert ["user", "tool_call", "tool_result", "tool_call", "approval_request", "approval", "text", "done"] == \
            [k for k in kinds(v) if k != "ready"]
        assert str(tmp_path) in p.user[0]

        last = v["last_seq"]
        c.post(f"/api/agent/chats/{cid}/messages", json={"text": "and now?"})
        wait(c, cid, lambda v: v["status"] == "idle" and v["last_seq"] > last + 2)
        new = c.get(f"/api/agent/chats/{cid}", params={"after": last}).json()["data"]["events"]
        assert [e["kind"] for e in new] == ["user", "text", "done"] and new[1]["data"]["text"] == "Second answer."
        assert p.user == [p.user[0], "and now?"]  # same history: the model sees both messages
        assert c.get("/api/agent/chats").json()["data"][0]["title"] == "train a churn model"
        assert c.delete(f"/api/agent/chats/{cid}").json()["ok"]
        assert c.get(f"/api/agent/chats/{cid}").status_code == 404


def test_stop_while_waiting_for_approval(tmp_path):
    plan = {"task": {"task_type": "toy", "input_modality": "x", "output_modality": "y"},
            "data_path": str(tmp_path), "pack": "none", "recipe": "none"}
    p = Scripted([call("run_start", plan=plan), say("never reached")])
    with make(p) as c:
        cid = c.post("/api/agent/chats", json={"brain": "anthropic:m"}).json()["data"]["id"]
        wait(c, cid, lambda v: v["status"] == "idle")
        c.post(f"/api/agent/chats/{cid}/messages", json={"text": "go"})
        wait(c, cid, lambda v: v["status"] == "approval")
        c.post(f"/api/agent/chats/{cid}/stop")
        v = wait(c, cid, lambda v: v["status"] == "idle" and "done" in kinds(v))
        done = [e for e in v["events"] if e["kind"] == "done"][-1]
        assert done["data"]["stop"] == "stopped" and len(p.script) == 1  # the model was not called again
        r = c.post(f"/api/agent/chats/{cid}/approval", json={"approved": True})
        assert r.status_code == 409


def test_cloud_brain_cannot_read_sensitive_data_from_the_gui(tmp_path):
    data = tmp_path / "patients.csv"
    data.write_text("age,diagnosis\n40,x\n")
    registry.mark(str(data), Sensitivity.sensitive)
    p = Scripted([call("data_profile", path=str(data)), say("I cannot use this data.")])
    with make(p) as c:
        cid = c.post("/api/agent/chats", json={"brain": "anthropic:m"}).json()["data"]["id"]
        wait(c, cid, lambda v: v["status"] == "idle")
        c.post(f"/api/agent/chats/{cid}/messages", json={"text": "train on it"})
        v = wait(c, cid, lambda v: v["status"] == "idle" and "done" in kinds(v))
        result = next(e for e in v["events"] if e["kind"] == "tool_result")
        assert result["data"]["ok"] is False and result["data"]["error"] == "PRIVACY_DENIED"
    # a local brain is allowed (same page, same data)
    p2 = Scripted([call("data_profile", path=str(data)), say("profiled")])
    with make(p2) as c:
        chat = c.post("/api/agent/chats", json={"brain": "ollama:qwen3"}).json()["data"]
        assert chat["local"] is True and chat["actor"] == "ai:local:ollama:qwen3"
        wait(c, chat["id"], lambda v: v["status"] == "idle")
        c.post(f"/api/agent/chats/{chat['id']}/messages", json={"text": "train on it"})
        v = wait(c, chat["id"], lambda v: v["status"] == "idle" and "done" in kinds(v))
        assert next(e for e in v["events"] if e["kind"] == "tool_result")["data"]["ok"] is True


def test_bad_brain_and_config():
    with make(Scripted([])) as c:
        r = c.post("/api/agent/chats", json={"brain": "nonsense"})
        assert r.status_code == 400 and r.json()["error"]["code"] == "PROVIDER_ERROR"
        cfg = c.get("/api/agent/config").json()["data"]
        assert cfg["default_brain"] and set(cfg["keys"]) == {"anthropic", "openai"}
        assert isinstance(cfg["ollama_models"], list)
        r = c.get("/api/agent/chats/nope")
        assert r.status_code == 404


def test_chat_page_in_a_real_browser(tmp_path):
    """The page: new conversation → message with data → approve in the page → run link and metrics → follow-up."""
    import glob
    import os
    import re
    import threading
    from pathlib import Path

    import pytest
    pw = pytest.importorskip("playwright.sync_api")
    import uvicorn

    fixture = Path(__file__).parent / "fixtures" / "engine-pack"
    data = tmp_path / "d.txt"
    data.write_text("x")
    plan = {"task": {"task_type": "toy", "input_modality": "x", "output_modality": "y", "metric": "loss"},
            "data_path": str(data), "pack": "engine-test", "recipe": "quick"}
    run_ids = []

    def wait_run(prev):
        run_ids.append(json.loads(prev[0].content)["data"]["run_id"])
        return Turn(tool_calls=[ToolCall("w", "run_wait", {"run_id": run_ids[0], "timeout_seconds": 60})], stop="tool_use")

    p = Scripted([call("hardware_profile"), call("pack_install", source=str(fixture)), call("run_start", plan=plan),
                  wait_run, say("訓練完成，loss 很低。"), say("可以，下一步建議自動調參。")])
    port = 8930 + os.getpid() % 60
    server = uvicorn.Server(uvicorn.Config(create_app("e2e", provider_factory=lambda spec: p),
                                           host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    exe = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))
    try:
        with pw.sync_playwright() as play:
            try:
                browser = play.chromium.launch(executable_path=exe[-1]) if exe else play.chromium.launch()
            except Exception as e:  # no browser in this environment
                pytest.skip(f"chromium not available: {e}")
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            expect = pw.expect
            page.goto(f"http://127.0.0.1:{port}/#/agent")
            expect(page.get_by_text("雲端模型：")).to_be_visible()
            page.click("#new-chat")
            page.wait_for_url(re.compile(r"#/agent/\w+"))
            expect(page.locator("#chat-status")).to_have_text("等你輸入")
            page.fill("#chat-input", "訓練一個模型")
            page.fill("#chat-data-path", str(data))
            page.click("#chat-send")
            expect(page.locator(".msg.approval pre")).to_contain_text("engine-test / quick", timeout=20_000)
            page.click(".msg.approval .approve")
            expect(page.get_by_text("訓練完成，loss 很低。")).to_be_visible(timeout=60_000)
            expect(page.locator(".msg.tool a[href^='#/runs/']").first).to_be_visible()
            expect(page.locator("#chat-status")).to_have_text("等你輸入")
            page.fill("#chat-input", "接下來呢？")
            page.keyboard.press("Control+Enter")
            expect(page.get_by_text("可以，下一步建議自動調參。")).to_be_visible(timeout=20_000)
            expect(page.get_by_text("nullnull")).to_have_count(0)
            page.set_viewport_size({"width": 390, "height": 800})  # phone width: no horizontal scroll
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
            assert errors == [], errors
            browser.close()
    finally:
        server.should_exit = True
    assert len(run_ids) == 1 and p.user[1] == "接下來呢？"
