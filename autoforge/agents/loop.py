"""Built-in agent loop (P4b): an LLM drives AutoForge through the same tools
the MCP server exposes. The tools run in-process via an MCP client, so the
privacy guard, audit log and error envelopes are identical to P4a.

Human approval gate: by default the person confirms before training starts
(`run_start`, `run_retry`); --yes turns it off. A declined call returns
USER_DECLINED with the person's feedback so the model can revise the plan.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from mcp import Client

from autoforge import config
from autoforge.agents.providers import Provider, ToolCall, ToolResult, ToolSpec
from autoforge.schemas.core import BrainKind
from autoforge.serve.mcp_server import INSTRUCTIONS, build_server

GATED_TOOLS = {"run_start", "run_retry", "hpo_start"}
MAX_RESULT_CHARS = 20_000

SYSTEM_PROMPT = """\
You are AutoForge's training agent. You analyse what the user needs, choose a training method that fits
their data and hardware, train the model on their machine, and report the result.

{instructions}

How to work:
- Start with hardware_profile and data_profile. Base every choice on what the tools return.
- Search packs for the task and read pack_info's knowledge before choosing a recipe; follow its guidance
  (e.g. start with the simple baseline recipe).
- Validate the plan, start the run, then poll with run_wait until it finishes. If it fails, read run_logs,
  fix the plan and try again. Compare against the baseline before trying anything heavier.
- If a tool returns USER_DECLINED, read the user's feedback in the message and revise.
- If a tool returns PRIVACY_DENIED, stop and explain; never look for another way to read that data.
- After a working baseline, use run_diagnose for problems and hpo_start to tune (trial 0 is the baseline);
  report whether tuning improved the metric.
- When the user wants to use the model, register the best run with model_register and show a
  model_predict example.
- Finish with a short report: the method and why, the metrics, the run_id and the model path.
"""


@dataclass
class AgentEvent:
    kind: str  # text | tool_call | tool_result | approval | info | done
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    final_text: str
    steps: int
    stop: str  # end | refusal | max_steps | error
    runs: list[str]
    transcript: str


ApproveFn = Callable[[ToolCall], Awaitable[tuple[bool, str]]]
EventFn = Callable[[AgentEvent], None]


def _truncate(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + f"\n... [truncated {len(text) - MAX_RESULT_CHARS} chars]"


class Transcript:
    """JSONL record of the whole agent session, next to the runs it started."""

    def __init__(self) -> None:
        d = config.home() / "agent-sessions"
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    def write(self, kind: str, **data: Any) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, **data}, ensure_ascii=False, default=str) + "\n")


def describe_call(call: ToolCall) -> str:
    """Plain-text summary of a gated call (what training would start), shown before asking the person."""
    if call.name == "run_retry":
        return f"retry run {call.arguments.get('run_id')}"
    if call.name == "hpo_start":
        a = call.arguments
        head = (f"  hyperparameter search: {a.get('n_trials', 10)} trials, optimise {a.get('metric')}"
                f" ({a.get('direction') or 'auto direction'})\n")
        return head + describe_call(ToolCall(call.id, "run_start", {"plan": a.get("plan")}))
    plan = call.arguments.get("plan") or {}
    task = plan.get("task") or {}
    method = (f"{plan.get('pack')} / {plan.get('recipe')}" if plan.get("pack")
              else f"custom code {plan.get('custom_code_path')}")
    lines = [
        f"  task:            {task.get('task_type')} (target: {task.get('target')}, metric: {task.get('metric')})",
        f"  data:            {plan.get('data_path')}",
        f"  method:          {method}",
        f"  device:          {plan.get('device', 'auto')}",
        f"  hyperparameters: {json.dumps(plan.get('hyperparameters') or {}, ensure_ascii=False)}",
    ]
    return "\n".join(lines)



class AgentSession:
    """A conversation with the agent: enter once, `send()` each user message, exit at the end.

    The MCP tools stay connected between messages and the provider keeps the whole history, so a
    follow-up like "now tune it" builds on what the agent already did. `should_stop` is checked
    between model turns (the GUI's stop button); a stopped turn ends with stop="stopped".
    """

    def __init__(self, provider: Provider, actor: str, brain: BrainKind, approve: ApproveFn | None = None,
                 on_event: EventFn | None = None, max_steps: int = 40,
                 should_stop: Callable[[], bool] | None = None, origin: str | None = None):
        self.provider, self.actor, self.brain = provider, actor, brain
        self.approve, self.max_steps = approve, max_steps
        self.emit = on_event or (lambda e: None)
        self.should_stop = should_stop or (lambda: False)
        self.origin = origin
        self.runs: list[str] = []
        self.transcript: Transcript | None = None
        self._client: Client | None = None

    async def __aenter__(self) -> "AgentSession":
        self.transcript = Transcript()
        self.transcript.write("start", actor=self.actor, brain=self.brain.value, origin=self.origin)
        server = build_server(self.actor, log_level="WARNING")  # the agent reports tool errors itself
        self._client = Client(server)
        await self._client.__aenter__()
        listed = await self._client.list_tools()
        tools = [ToolSpec(t.name, t.description or "", t.input_schema) for t in listed.tools]
        self.provider.start(SYSTEM_PROMPT.format(instructions=INSTRUCTIONS), tools)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.__aexit__(*exc)
            self._client = None
        if self.transcript:
            self.transcript.write("close", runs=self.runs)

    async def send(self, text: str, data_path: str | None = None) -> AgentResult:
        """One user message → model turns and tool calls until the model answers (or a limit is hit)."""
        assert self._client is not None and self.transcript is not None, "use `async with AgentSession(...)`"
        mcp, transcript, emit = self._client, self.transcript, self.emit
        user_msg = text + (f"\n\nData: {Path(data_path).expanduser().resolve()}" if data_path else "")
        transcript.write("user", text=text, data_path=data_path)
        self.provider.add_user(user_msg)
        runs: list[str] = []

        final_text, stop, step = "", "max_steps", 0
        for step in range(1, self.max_steps + 1):
            if self.should_stop():
                stop = "stopped"
                emit(AgentEvent("info", {"message": "stopped by the user"}))
                break
            turn = await self.provider.complete()
            if turn.text:
                final_text = turn.text
                emit(AgentEvent("text", {"text": turn.text}))
                transcript.write("text", text=turn.text)
            if turn.stop == "refusal":
                stop = "refusal"
                emit(AgentEvent("info", {"message": f"model declined: {turn.detail or ''}"}))
                break
            if not turn.tool_calls:
                if turn.stop == "pause":
                    continue  # server paused a long turn; resend to let it continue
                stop = "end"
                break

            results = []
            for call in turn.tool_calls:
                emit(AgentEvent("tool_call", {"name": call.name, "arguments": call.arguments}))
                transcript.write("tool_call", name=call.name, arguments=call.arguments)
                if call.parse_error:
                    content = json.dumps({"ok": False, "error": {"code": "BAD_TOOL_CALL", "message": call.parse_error}})
                    results.append(ToolResult(call.id, content, is_error=True))
                    continue
                if call.name in GATED_TOOLS and self.approve is not None:
                    ok, feedback = await self.approve(call)
                    emit(AgentEvent("approval", {"name": call.name, "approved": ok, "feedback": feedback}))
                    transcript.write("approval", name=call.name, approved=ok, feedback=feedback)
                    if not ok:
                        msg = "The user declined this action." + (f" Feedback: {feedback}" if feedback else "")
                        content = json.dumps({"ok": False, "error": {"code": "USER_DECLINED", "message": msg}})
                        results.append(ToolResult(call.id, content, is_error=True))
                        continue
                res = await mcp.call_tool(call.name, call.arguments)
                envelope = res.structured_content
                if envelope is None:  # unstructured (e.g. argument validation error from the MCP layer)
                    text_out = "\n".join(getattr(c, "text", "") for c in res.content)
                    envelope = {"ok": False, "error": {"code": "BAD_TOOL_CALL", "message": text_out}}
                if call.name in ("run_start", "run_retry") and envelope.get("ok"):
                    runs.append(envelope["data"]["run_id"])
                if call.name == "hpo_start" and envelope.get("ok"):
                    runs.append(envelope["data"]["study_id"])
                content = _truncate(json.dumps(envelope, ensure_ascii=False, default=str))
                emit(AgentEvent("tool_result", {"name": call.name, "ok": envelope.get("ok"),
                                                "error": (envelope.get("error") or {}).get("code"),
                                                "data": _brief(call.name, envelope)}))
                transcript.write("tool_result", name=call.name, ok=envelope.get("ok"), content=content[:2000])
                results.append(ToolResult(call.id, content, is_error=not envelope.get("ok", False)))
            self.provider.add_tool_results(results)

        self.runs += runs
        transcript.write("done", stop=stop, steps=step, runs=runs)
        emit(AgentEvent("done", {"stop": stop, "steps": step, "runs": runs}))
        return AgentResult(final_text=final_text, steps=step, stop=stop, runs=runs, transcript=str(transcript.path))


def _brief(name: str, envelope: dict) -> dict[str, Any] | None:
    """The few result fields a UI shows next to a tool call (ids and metrics), never whole payloads."""
    data = envelope.get("data")
    if not envelope.get("ok") or not isinstance(data, dict):
        return None
    out = {}
    for k in ("run_id", "study_id", "status", "metrics", "model", "version", "name", "best_value", "improvement"):
        v = data.get(k)
        if isinstance(v, (int, float, bool, dict)) or (isinstance(v, str) and len(v) < 200):
            out[k] = v
    return out or None


async def run_agent(
    goal: str,
    provider: Provider,
    actor: str,
    brain: BrainKind,
    approve: ApproveFn | None = None,
    on_event: EventFn | None = None,
    max_steps: int = 40,
    data_path: str | None = None,
) -> AgentResult:
    """One-shot agent (CLI `autoforge train`): a single goal, run until the model reports back."""
    async with AgentSession(provider, actor, brain, approve, on_event, max_steps, origin="cli") as session:
        return await session.send(goal, data_path)
