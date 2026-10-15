import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS

import httpx2
import pytest

from autoforge.agents.loop import run_agent
from autoforge.agents.providers import (
    AnthropicProvider, OpenAICompatProvider, Provider, ProviderError, ToolCall, ToolResult, ToolSpec, Turn, parse_brain,
)
from autoforge.data import registry
from autoforge.schemas.core import BrainKind, Sensitivity

FIXTURE = Path(__file__).parent / "fixtures" / "engine-pack"


class ScriptedProvider(Provider):
    """A fake LLM: `script` is a list of functions (last_results) -> Turn."""

    def __init__(self, script):
        self.script = list(script)
        self.results: list[list[ToolResult]] = []
        self.tools: list[ToolSpec] = []
        self.user: list[str] = []

    def start(self, system, tools):
        self.system, self.tools = system, tools

    def add_user(self, text):
        self.user.append(text)

    async def complete(self):
        step = self.script.pop(0)
        return step(self.results[-1] if self.results else [])

    def add_tool_results(self, results):
        self.results.append(results)

    def envelope(self, i=-1, j=0):
        return json.loads(self.results[i][j].content)


def call(name, **args):
    return lambda _: Turn(tool_calls=[ToolCall(f"c-{name}", name, args)], stop="tool_use")


def say(text):
    return lambda _: Turn(text=text, stop="end")


def go(provider, actor="ai:test", approve=None, max_steps=40, data_path=None):
    events = []
    result = asyncio.run(run_agent("train something", provider, actor=actor, brain=BrainKind.api,
                                   approve=approve, on_event=events.append, max_steps=max_steps,
                                   data_path=data_path))
    return result, events


@pytest.fixture
def data(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("x")
    return p


def plan(data, **kw):
    return {"task": {"task_type": "toy", "input_modality": "x", "output_modality": "y", "metric": "loss"},
            "data_path": str(data), "pack": "engine-test", "recipe": "quick"} | kw


def test_full_flow_with_approval(data):
    run_ids = []

    def wait(prev):
        run_ids.append(json.loads(prev[0].content)["data"]["run_id"])
        return Turn(tool_calls=[ToolCall("w", "run_wait", {"run_id": run_ids[0], "timeout_seconds": 60})], stop="tool_use")

    p = ScriptedProvider([call("hardware_profile"), call("pack_install", source=str(FIXTURE)),
                          call("run_start", plan=plan(data)), wait, say("Done: accuracy 0.9")])
    asked = []

    async def approve(c):
        asked.append(c.name)
        return True, ""

    result, events = go(p, approve=approve, data_path=str(data))
    assert result.stop == "end" and result.final_text == "Done: accuracy 0.9"
    assert result.runs == run_ids and asked == ["run_start"]
    assert p.envelope()["data"]["status"] == "succeeded"
    assert str(data.resolve()) in p.user[0]
    assert {t.name for t in p.tools} >= {"run_start", "data_profile", "pack_search"}
    assert Path(result.transcript).is_file()
    assert events[-1].kind == "done"


def test_declined_with_feedback_reaches_the_model(data):
    p = ScriptedProvider([call("pack_install", source=str(FIXTURE)), call("run_start", plan=plan(data)), say("ok")])

    async def approve(c):
        return False, "use fewer steps"

    result, _ = go(p, approve=approve)
    env = p.envelope()
    assert env["error"]["code"] == "USER_DECLINED" and "use fewer steps" in env["error"]["message"]
    assert p.results[-1][0].is_error and result.runs == []


def test_no_approval_when_yes(data):
    p = ScriptedProvider([call("pack_install", source=str(FIXTURE)), call("run_start", plan=plan(data)), say("ok")])
    result, _ = go(p, approve=None)
    assert len(result.runs) == 1


def test_bad_tool_input_is_reported_not_run():
    bad = lambda _: Turn(tool_calls=[ToolCall("x", "run_status", {}, parse_error="truncated")], stop="tool_use")
    p = ScriptedProvider([bad, call("run_status"), say("ok")])
    go(p)
    assert p.envelope(0)["error"]["code"] == "BAD_TOOL_CALL"
    assert p.envelope(1)["ok"] is False  # missing run_id rejected by the MCP layer, as an envelope


def test_refusal_and_max_steps():
    r, _ = go(ScriptedProvider([lambda _: Turn(stop="refusal", detail="cyber: no")]))
    assert r.stop == "refusal"
    r, _ = go(ScriptedProvider([call("pack_list")] * 3), max_steps=3)
    assert r.stop == "max_steps" and r.steps == 3


def test_pause_turn_continues():
    p = ScriptedProvider([lambda _: Turn(stop="pause"), say("finished")])
    r, _ = go(p)
    assert r.stop == "end" and r.final_text == "finished"


def test_privacy_cloud_agent_cannot_read_sensitive_data(tmp_path):
    secret = tmp_path / "s.csv"
    secret.write_text("a,b\n1,2\n")
    registry.mark(secret, Sensitivity.sensitive)
    p = ScriptedProvider([call("data_profile", path=str(secret)), say("stopped")])
    go(p, actor="ai:anthropic:claude-opus-5-5")
    assert p.envelope()["error"]["code"] == "PRIVACY_DENIED"
    p = ScriptedProvider([call("data_profile", path=str(secret)), say("ok")])
    go(p, actor="ai:local:ollama:qwen3")
    assert p.envelope()["ok"] is True


# ---------------- brain specs

def test_parse_brain(monkeypatch):
    monkeypatch.delenv("AUTOFORGE_BRAIN", raising=False)
    b = parse_brain(None)
    assert (b.provider, b.model, b.brain_kind) == ("anthropic", "claude-opus-5-5", BrainKind.api)
    o = parse_brain("ollama:qwen3")
    assert o.brain_kind is BrainKind.local and o.actor == "ai:local:ollama:qwen3"
    assert parse_brain("openai:gpt-x").actor == "ai:openai:gpt-x"
    for bad in ["nocolon", "weird:model"]:
        with pytest.raises(ProviderError):
            parse_brain(bad)
    with pytest.raises(ProviderError):
        parse_brain("local:m")  # needs --base-url
    monkeypatch.setenv("AUTOFORGE_BRAIN", "ollama:llama3")
    assert parse_brain(None).model == "llama3"


# ---------------- Claude adapter (stub SDK client)

class StubMessages:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    async def create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


def stub_client(*responses):
    msgs = StubMessages(responses)
    return NS(beta=NS(messages=msgs)), msgs


def test_anthropic_request_shape_and_tool_round_trip():
    thinking = NS(type="thinking", thinking="", signature="sig")
    tool_use = NS(type="tool_use", id="tu1", name="run_status", input={"run_id": "r"})
    r1 = NS(content=[thinking, NS(type="text", text="checking"), tool_use], stop_reason="tool_use", stop_details=None)
    r2 = NS(content=[NS(type="text", text="all good")], stop_reason="end_turn", stop_details=None)
    client, msgs = stub_client(r1, r2)
    p = AnthropicProvider("claude-opus-5-5", client=client)
    p.start("sys", [ToolSpec("run_status", "d", {"type": "object"})])
    p.add_user("hi")
    t = asyncio.run(p.complete())
    assert t.text == "checking" and t.tool_calls[0].arguments == {"run_id": "r"} and t.stop == "tool_use"
    kw = msgs.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"] == {"effort": "high"}
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["tools"][0]["input_schema"] == {"type": "object"} and kw["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in kw and "thinking" not in kw
    p.add_tool_results([ToolResult("tu1", '{"ok": true}')])
    assert p.messages[1]["content"] is r1.content  # assistant content echoed back unchanged (thinking kept)
    assert p.messages[2]["content"][0] == {"type": "tool_result", "tool_use_id": "tu1",
                                           "content": '{"ok": true}', "is_error": False}
    assert asyncio.run(p.complete()).stop == "end"


def test_anthropic_no_fallbacks_for_other_models_and_refusal_detail():
    r = NS(content=[], stop_reason="refusal", stop_details=NS(category="cyber", explanation="nope"))
    client, msgs = stub_client(r)
    p = AnthropicProvider("claude-haiku-5-5", client=client)
    p.start("s", [])
    p.add_user("x")
    t = asyncio.run(p.complete())
    assert t.stop == "refusal" and t.detail == "cyber: nope"
    assert "fallbacks" not in msgs.calls[0] and "betas" not in msgs.calls[0]


def test_anthropic_max_tokens_marks_tool_calls_unusable():
    r = NS(content=[NS(type="tool_use", id="t", name="x", input={})], stop_reason="max_tokens", stop_details=None)
    client, _ = stub_client(r)
    p = AnthropicProvider("claude-opus-5-5", client=client)
    p.start("s", [])
    t = asyncio.run(p.complete())
    assert t.stop == "max_tokens" and t.tool_calls[0].parse_error


def test_anthropic_api_error_becomes_provider_error():
    class Boom:
        async def create(self, **kw):
            raise RuntimeError("rate limited")
    p = AnthropicProvider("claude-opus-5-5", client=NS(beta=NS(messages=Boom())))
    p.start("s", [])
    with pytest.raises(ProviderError):
        asyncio.run(p.complete())


# ---------------- OpenAI-compatible adapter (mock HTTP)

def test_openai_compat_tool_calls_and_results():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "run_status", "arguments": '{"run_id": "r1"}'}},
                {"id": "b", "type": "function", "function": {"name": "pack_list", "arguments": "{bad json"}},
            ]}
            return httpx2.Response(200, json={"choices": [{"message": msg, "finish_reason": "tool_calls"}]})
        return httpx2.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "done"},
                                                       "finish_reason": "stop"}]})

    p = OpenAICompatProvider("qwen3", "http://localhost:11434/v1", None, transport=httpx2.MockTransport(handler))
    p.start("sys", [ToolSpec("run_status", "d", {"type": "object"})])
    p.add_user("go")
    t = asyncio.run(p.complete())
    assert [c.name for c in t.tool_calls] == ["run_status", "pack_list"]
    assert t.tool_calls[0].arguments == {"run_id": "r1"} and t.tool_calls[1].parse_error
    assert seen[0]["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert seen[0]["messages"][0] == {"role": "system", "content": "sys"}
    p.add_tool_results([ToolResult("a", "x"), ToolResult("b", "y", True)])
    t2 = asyncio.run(p.complete())
    assert t2.text == "done" and t2.stop == "end"
    assert seen[1]["messages"][-2:] == [{"role": "tool", "tool_call_id": "a", "content": "x"},
                                        {"role": "tool", "tool_call_id": "b", "content": "y"}]


def test_openai_compat_errors():
    p = OpenAICompatProvider("m", "http://x/v1", None,
                             transport=httpx2.MockTransport(lambda r: httpx2.Response(404, text="model not found")))
    p.start("s", [])
    with pytest.raises(ProviderError) as e:
        asyncio.run(p.complete())
    assert "404" in str(e.value)

    def refuse(request):
        raise httpx2.ConnectError("refused")
    p = OpenAICompatProvider("m", "http://x/v1", None, transport=httpx2.MockTransport(refuse))
    p.start("s", [])
    with pytest.raises(ProviderError) as e:
        asyncio.run(p.complete())
    assert "ollama serve" in e.value.hint
