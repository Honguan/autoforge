"""AI agent chat for the GUI: the built-in agent (P4b), one conversation per ChatSession.

Each conversation runs in its own thread with its own event loop, so a slow model call or a
`run_wait` tool call never blocks the web server. The page polls `events(after=seq)`.

- The agent acts as its brain (actor `ai:<provider>:<model>` or `ai:local:...`), not as the
  human who opened the page: the privacy guard and audit log treat it exactly like the CLI agent.
- Gated tools (run_start, run_retry, hpo_start) wait for the person's answer in the page.
- Stop ends the agent's turn between model calls; training runs it already started keep going
  (stop them on the run page).
"""

from __future__ import annotations

import asyncio
import queue
import secrets
import threading
import time
from typing import Any

from autoforge.agents.loop import AgentEvent, AgentSession, describe_call
from autoforge.agents.providers import BrainSpec, Provider, ToolCall
from autoforge.errors import AutoForgeError

MAX_SESSIONS = 20
MAX_EVENTS = 2000


class ChatError(AutoForgeError):
    code = "CHAT_ERROR"


class ChatNotFound(AutoForgeError):
    code = "NOT_FOUND"
    hint = "Start a new conversation."


class ChatSession:
    def __init__(self, spec: BrainSpec, provider: Provider, max_steps: int = 40):
        self.id = secrets.token_hex(6)
        self.spec = spec
        self.created = time.time()
        self.status = "starting"  # starting | idle | thinking | approval | closed | error
        self.pending: dict[str, Any] | None = None
        self.title = ""
        self.runs: list[str] = []
        self._events: list[dict[str, Any]] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._inbox: queue.Queue[tuple[str, str | None] | None] = queue.Queue()
        self._answers: queue.Queue[tuple[bool, str]] = queue.Queue()
        self._stop = threading.Event()
        self._provider, self._max_steps = provider, max_steps
        self._thread = threading.Thread(target=self._main, name=f"agent-chat-{self.id}", daemon=True)
        self._thread.start()

    # ---- called from the web server ----

    def send(self, text: str, data_path: str | None = None) -> None:
        text = text.strip()
        if not text:
            raise ChatError("empty message")
        if self.status != "idle":
            raise ChatError(f"the agent is {self.status}; wait for it (or press stop) before sending",
                            hint="poll the conversation until status is 'idle'")
        self.title = self.title or text[:60]
        self._stop.clear()
        self.status = "thinking"
        self._push("user", {"text": text, "data_path": data_path})
        self._inbox.put((text, data_path))

    def answer(self, approved: bool, feedback: str = "") -> None:
        if self.pending is None:
            raise ChatError("nothing is waiting for approval")
        self.pending = None
        self._answers.put((bool(approved), feedback.strip()))

    def stop(self) -> None:
        self._stop.set()
        if self.pending is not None:
            self.answer(False, "stopped by the user")

    def close(self) -> None:
        self.stop()
        self._inbox.put(None)

    def view(self, after: int = 0) -> dict[str, Any]:
        with self._lock:
            events = [e for e in self._events if e["seq"] > after]
        return {"id": self.id, "status": self.status, "pending": self.pending, "events": events,
                "last_seq": self._seq, **self.summary()}

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title or "（新對話）", "status": self.status, "created": self.created,
                "brain": f"{self.spec.provider}:{self.spec.model}", "actor": self.spec.actor,
                "local": self.spec.is_local, "runs": list(self.runs)}

    # ---- the conversation thread ----

    def _push(self, kind: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            self._events.append({"seq": self._seq, "t": time.time(), "kind": kind, "data": data})
            del self._events[:-MAX_EVENTS]

    def _on_event(self, event: AgentEvent) -> None:
        if event.kind == "done":
            self.runs += [r for r in event.data.get("runs", []) if r not in self.runs]
        self._push(event.kind, event.data)

    async def _approve(self, call: ToolCall) -> tuple[bool, str]:
        self.pending = {"tool": call.name, "summary": describe_call(call), "arguments": call.arguments}
        self.status = "approval"
        self._push("approval_request", {"tool": call.name, "summary": self.pending["summary"]})
        approved, feedback = await asyncio.to_thread(self._answers.get)
        self.status = "thinking"
        return approved, feedback

    def _main(self) -> None:
        try:
            asyncio.run(self._loop())
        except Exception as e:  # noqa: BLE001 — shown to the person, the thread must not die silently
            self._push("error", {"message": f"{type(e).__name__}: {e}"})
            self.status = "error"

    async def _loop(self) -> None:
        async with AgentSession(self._provider, self.spec.actor, self.spec.brain_kind, approve=self._approve,
                                on_event=self._on_event, max_steps=self._max_steps,
                                should_stop=self._stop.is_set, origin="gui") as session:
            self._push("ready", {"brain": f"{self.spec.provider}:{self.spec.model}", "actor": self.spec.actor,
                                 "local": self.spec.is_local})
            self.status = "idle"
            while True:
                msg = await asyncio.to_thread(self._inbox.get)
                if msg is None:
                    break
                text, data_path = msg
                try:
                    await session.send(text, data_path)
                except AutoForgeError as e:  # e.g. the provider rejected the API key
                    self._push("error", {"message": str(e), "code": e.code, "hint": e.hint})
                except Exception as e:  # noqa: BLE001
                    self._push("error", {"message": f"{type(e).__name__}: {e}"})
                self.pending = None
                self.status = "idle"
        self.status = "closed"


class ChatManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()

    def create(self, spec: BrainSpec, provider: Provider, max_steps: int = 40) -> ChatSession:
        with self._lock:
            if len(self._sessions) >= MAX_SESSIONS:  # drop the oldest closed/idle conversation
                victims = sorted((s for s in self._sessions.values() if s.status in ("idle", "closed", "error")),
                                 key=lambda s: s.created)
                if not victims:
                    raise ChatError(f"{MAX_SESSIONS} conversations are busy", hint="stop or close one first")
                victims[0].close()
                del self._sessions[victims[0].id]
            session = ChatSession(spec, provider, max_steps)
            self._sessions[session.id] = session
            return session

    def get(self, chat_id: str) -> ChatSession:
        try:
            return self._sessions[chat_id]
        except KeyError:
            raise ChatNotFound(f"no conversation '{chat_id}'")

    def list(self) -> list[dict[str, Any]]:
        return [s.summary() for s in sorted(self._sessions.values(), key=lambda s: s.created, reverse=True)]

    def close(self, chat_id: str) -> None:
        self.get(chat_id).close()
        with self._lock:
            self._sessions.pop(chat_id, None)

    def close_all(self) -> None:
        for chat_id in list(self._sessions):
            self.close(chat_id)
