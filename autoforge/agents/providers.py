"""LLM providers for the built-in agent (P4b).

Each provider owns its own conversation state, so the agent loop stays
provider-neutral:
    p.start(system, tools); p.add_user(text)
    turn = await p.complete()          # text + tool calls
    p.add_tool_results([...])          # one result per call, all at once

Brain specs:
    anthropic:<model>   Claude via the official SDK        (cloud)
    openai:<model>      OpenAI or any cloud OpenAI-compatible API (--base-url)
    ollama:<model>      local Ollama (OpenAI-compatible endpoint)
    local:<model>       any local OpenAI-compatible server (vLLM, LM Studio, llama.cpp) with --base-url
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from autoforge.errors import AutoForgeError
from autoforge.schemas.core import BrainKind

DEFAULT_BRAIN = "anthropic:claude-opus-5-5"
# Models that accept server-side refusal fallbacks (fallbacks: "default").
CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class ProviderError(AutoForgeError):
    code = "PROVIDER_ERROR"


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str | None = None


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop: str = "end"  # end | tool_use | pause | max_tokens | refusal
    detail: str | None = None


@dataclass
class BrainSpec:
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None

    @property
    def is_local(self) -> bool:
        return self.provider in ("ollama", "local")

    @property
    def brain_kind(self) -> BrainKind:
        return BrainKind.local if self.is_local else BrainKind.api

    @property
    def actor(self) -> str:
        # 'ai:local:...' is what the privacy guard recognises as a local model.
        return f"ai:local:{self.provider}:{self.model}" if self.is_local else f"ai:{self.provider}:{self.model}"


def parse_brain(spec: str | None, base_url: str | None = None, api_key_env: str | None = None) -> BrainSpec:
    spec = spec or os.environ.get("AUTOFORGE_BRAIN") or DEFAULT_BRAIN
    provider, sep, model = spec.partition(":")
    if not sep or not model:
        raise ProviderError(f"bad brain '{spec}'", hint="Use provider:model, e.g. anthropic:claude-opus-5-5 or ollama:qwen3")
    if provider not in ("anthropic", "openai", "ollama", "local"):
        raise ProviderError(f"unknown provider '{provider}'", hint="Choose anthropic, openai, ollama or local.")
    if provider == "local" and not base_url:
        raise ProviderError("local:<model> needs --base-url (an OpenAI-compatible endpoint, e.g. http://localhost:8000/v1)")
    return BrainSpec(provider, model, base_url, api_key_env)


def make_provider(spec: BrainSpec) -> "Provider":
    if spec.provider == "anthropic":
        return AnthropicProvider(spec.model)
    if spec.provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if not host.startswith("http"):
            host = "http://" + host
        return OpenAICompatProvider(spec.model, spec.base_url or f"{host}/v1", api_key=None)
    if spec.provider == "local":
        return OpenAICompatProvider(spec.model, spec.base_url, api_key=None)
    key_env = spec.api_key_env or "OPENAI_API_KEY"
    key = os.environ.get(key_env)
    if not key:
        raise ProviderError(f"{key_env} is not set", hint=f"export {key_env}=... (or pass --api-key-env)")
    return OpenAICompatProvider(spec.model, spec.base_url or "https://api.openai.com/v1", api_key=key)


class Provider:
    def start(self, system: str, tools: list[ToolSpec]) -> None: ...
    def add_user(self, text: str) -> None: ...
    async def complete(self) -> Turn: ...
    def add_tool_results(self, results: list[ToolResult]) -> None: ...


# ---------------------------------------------------------------- Claude


class AnthropicProvider(Provider):
    """Claude via the official Anthropic SDK (manual tool-use loop).

    - Effort 'high' (agentic work); adaptive thinking is the model default.
    - Server-side refusal fallback ('default') on models that support it.
    - The assistant content is appended back unchanged (thinking blocks included),
      so the history stays append-only.
    """

    def __init__(self, model: str, client: Any = None, effort: str = "high", max_tokens: int = 16000):
        if client is None:
            try:
                import anthropic
            except ImportError:
                raise ProviderError("the anthropic package is not installed", hint='pip install "autoforge[anthropic]"')
            client = anthropic.AsyncAnthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = []

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        self.system = system
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    async def complete(self) -> Turn:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},  # tools + system + history are a stable prefix
        )
        if self.model in CLAUDE_FALLBACK_MODELS:
            kwargs |= {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        try:
            response = await self.client.beta.messages.create(**kwargs)
        except Exception as e:  # SDK errors: auth, rate limit, network, bad request
            raise ProviderError(f"Claude API call failed: {type(e).__name__}: {e}") from e

        self.messages.append({"role": "assistant", "content": response.content})
        text = "\n".join(b.text for b in response.content if b.type == "text" and b.text)
        calls = [ToolCall(b.id, b.name, dict(b.input or {})) for b in response.content if b.type == "tool_use"]
        stop = {"end_turn": "end", "tool_use": "tool_use", "pause_turn": "pause",
                "max_tokens": "max_tokens", "refusal": "refusal"}.get(response.stop_reason, "end")
        detail = None
        if stop == "refusal" and getattr(response, "stop_details", None):
            detail = f"{response.stop_details.category}: {response.stop_details.explanation}"
        if stop == "max_tokens":
            # A tool_use block cut off by max_tokens may have incomplete input: don't run it,
            # but every tool_use still needs a tool_result, so report it as an error.
            for c in calls:
                c.parse_error = "response hit max_tokens; tool input may be truncated, call the tool again"
        return Turn(text=text, tool_calls=calls, stop=stop, detail=detail)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error}
                for r in results
            ],
        })


# ---------------------------------------------------------------- OpenAI-compatible (Ollama, OpenAI, vLLM ...)


class OpenAICompatProvider(Provider):
    """Chat Completions with function calling over plain HTTP (httpx2)."""

    def __init__(self, model: str, base_url: str, api_key: str | None, transport: Any = None, timeout: float = 600):
        import httpx2

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx2.AsyncClient(base_url=base_url.rstrip("/") + "/", headers=headers,
                                         timeout=timeout, transport=transport)
        self.model = model
        self.base_url = base_url
        self.messages: list[dict[str, Any]] = []

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        self.messages = [{"role": "system", "content": system}]
        self.tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
            for t in tools
        ]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    async def complete(self) -> Turn:
        import httpx2

        body = {"model": self.model, "messages": self.messages, "tools": self.tools}
        try:
            r = await self.client.post("chat/completions", json=body)
            r.raise_for_status()
            data = r.json()
        except httpx2.HTTPStatusError as e:
            raise ProviderError(f"{self.base_url} returned {e.response.status_code}: {e.response.text[:500]}") from e
        except httpx2.RequestError as e:
            raise ProviderError(f"cannot reach {self.base_url}: {e}",
                                hint="Is the model server running? For Ollama: `ollama serve` and `ollama pull <model>`.") from e
        choice = data["choices"][0]
        msg = choice["message"]
        self.messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls") and v is not None})
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            raw = fn.get("arguments") or "{}"
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw)
                err = None
            except json.JSONDecodeError as e:
                args, err = {}, f"arguments are not valid JSON: {e}"
            calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args, err))
        finish = choice.get("finish_reason")
        stop = "tool_use" if calls else ("max_tokens" if finish == "length" else "end")
        return Turn(text=msg.get("content") or "", tool_calls=calls, stop=stop)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": r.content})
