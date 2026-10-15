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

GATED_TOOLS = {"run_start", "run_retry"}
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
    emit = on_event or (lambda e: None)
    transcript = Transcript()
    runs: list[str] = []
    server = build_server(actor)

    user_msg = goal
    if data_path:
        user_msg += f"\n\nData: {Path(data_path).expanduser().resolve()}"
    transcript.write("start", goal=goal, actor=actor, brain=brain.value, data_path=data_path)

    async with Client(server) as mcp:
        listed = await mcp.list_tools()
        tools = [ToolSpec(t.name, t.description or "", t.input_schema) for t in listed.tools]
        provider.start(SYSTEM_PROMPT.format(instructions=INSTRUCTIONS), tools)
        provider.add_user(user_msg)

        final_text, stop = "", "max_steps"
        for step in range(1, max_steps + 1):
            turn = await provider.complete()
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
                if call.name in GATED_TOOLS and approve is not None:
                    ok, feedback = await approve(call)
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
                    text = "\n".join(getattr(c, "text", "") for c in res.content)
                    envelope = {"ok": False, "error": {"code": "BAD_TOOL_CALL", "message": text}}
                if call.name in ("run_start", "run_retry") and envelope.get("ok"):
                    runs.append(envelope["data"]["run_id"])
                content = _truncate(json.dumps(envelope, ensure_ascii=False, default=str))
                emit(AgentEvent("tool_result", {"name": call.name, "ok": envelope.get("ok"),
                                                "error": (envelope.get("error") or {}).get("code")}))
                transcript.write("tool_result", name=call.name, ok=envelope.get("ok"), content=content[:2000])
                results.append(ToolResult(call.id, content, is_error=not envelope.get("ok", False)))
            provider.add_tool_results(results)
        else:
            step = max_steps

    transcript.write("done", stop=stop, steps=step, runs=runs)
    emit(AgentEvent("done", {"stop": stop, "steps": step, "runs": runs}))
    return AgentResult(final_text=final_text, steps=step, stop=stop, runs=runs, transcript=str(transcript.path))
