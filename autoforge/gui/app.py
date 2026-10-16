"""Local web GUI (P8): `autoforge gui`.

A Starlette app serving a static single-page UI and a /api/* JSON API that calls the same
core API as the CLI and MCP server (the GUI user acts as "human"). The AI assistant page
(/api/agent/*, see agent_chat.py) runs the built-in agent, which acts as its own brain.

Security (it can start training and run code on this machine):
- binds to 127.0.0.1 by default;
- every /api call must carry the per-launch token (header X-AutoForge-Token), which only the
  page opened from the printed URL knows — blocks other websites calling the API (CSRF);
- the Host header must be localhost/127.0.0.1 — blocks DNS-rebinding attacks.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
import re
import secrets
import urllib.request
from pathlib import Path
from typing import Any, Awaitable, Callable

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from autoforge import __version__, api, config
from autoforge.agents.providers import DEFAULT_BRAIN, BrainSpec, Provider, make_provider, parse_brain
from autoforge.errors import AutoForgeError
from autoforge.gui.agent_chat import ChatManager
from autoforge.schemas.core import BrainKind

ACTOR = "human"
BRAIN = BrainKind.human
STATIC = Path(__file__).parent / "static"
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
UPLOAD_MAX_BYTES = 500 * 1024 * 1024
STATUS = {"CHAT_ERROR": 409, "NOT_FOUND": 404, "RUN_NOT_FOUND": 404, "MODEL_NOT_FOUND": 404, "STUDY_NOT_FOUND": 404,
          "PACK_NOT_FOUND": 404, "INVALID_INPUT": 422, "PLAN_INVALID": 422, "UNAUTHORIZED": 401}


def uploads_dir() -> Path:
    return config.home() / "workspace" / "data"


def _ok(data: Any) -> JSONResponse:
    return JSONResponse({"ok": True, "data": data, "error": None})


def _err(code: str, message: str, hint: str | None = None, details: Any = None, status: int | None = None) -> JSONResponse:
    body = {"ok": False, "data": details, "error": {"code": code, "message": message, "hint": hint}}
    return JSONResponse(body, status_code=status or STATUS.get(code, 400))


Handler = Callable[[Request], Awaitable[Any]]


def endpoint(fn: Handler) -> Callable[[Request], Awaitable[Response]]:
    async def wrapped(request: Request) -> Response:
        try:
            result = await fn(request)
            return result if isinstance(result, Response) else _ok(result)
        except AutoForgeError as e:
            return _err(e.code, str(e), e.hint, e.details)
        except (ValueError, KeyError, TypeError) as e:
            return _err("BAD_REQUEST", f"{type(e).__name__}: {e}")
    return wrapped


async def _json(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError:
        raise api.BadRequest("body must be JSON")
    if not isinstance(body, dict):
        raise api.BadRequest("body must be a JSON object")
    return body


# ---------------------------------------------------------------- handlers

async def info(r):
    return {"version": __version__, "home": str(config.home()), "actor": ACTOR}

async def hardware(r):
    return api.hardware()

async def tasks(r):
    return api.task_catalog()

async def data_profile(r):
    return api.data_profile(r.query_params["path"], BRAIN)

async def data_files(r):
    d = uploads_dir()
    files = sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if d.is_dir() else []
    return [{"name": p.name, "path": str(p), "bytes": p.stat().st_size} for p in files if p.is_file()]

async def data_upload(r):
    form = await r.form(max_part_size=UPLOAD_MAX_BYTES)
    f = form.get("file")
    if f is None or not getattr(f, "filename", None):
        raise api.BadRequest("send a file in the 'file' field")
    name = re.sub(r"[^\w.\-]+", "_", Path(f.filename).name).strip("._") or "data.csv"
    d = uploads_dir()
    d.mkdir(parents=True, exist_ok=True)
    dest = d / name
    if dest.exists():
        dest = d / f"{dest.stem}-{secrets.token_hex(2)}{dest.suffix}"
    dest.write_bytes(await f.read())
    return {"name": dest.name, "path": str(dest), "bytes": dest.stat().st_size}

async def packs_search(r):
    q = r.query_params
    return api.pack_search(q.get("q") or None, q.get("task") or None, q.get("fit", "1") == "1")

async def packs_installed(r):
    return api.pack_list()

async def pack_detail(r):
    return api.pack_info(r.path_params["name"])

async def pack_install(r):
    body = await _json(r)
    return api.pack_install(body["source"], bool(body.get("force")))

async def plan_validate(r):
    return api.plan_validate((await _json(r))["plan"], ACTOR, BRAIN)

async def runs_list(r):
    return api.run_list(int(r.query_params.get("limit", 50)))

async def run_start(r):
    return api.run_start((await _json(r))["plan"], ACTOR, BRAIN)

async def run_detail(r):
    rid = r.path_params["run_id"]
    from autoforge.engine.store import RunStore
    rec = RunStore.open(rid).record()
    return api.run_status(rid) | {"plan": rec.plan.model_dump(mode="json"), "study_id": rec.study_id}

async def run_series(r):
    return api.run_series(r.path_params["run_id"])

async def run_logs(r):
    return api.run_logs(r.path_params["run_id"], int(r.query_params.get("tail", 200)))

async def run_audit(r):
    return api.run_audit(r.path_params["run_id"])

async def run_diagnose(r):
    return api.run_diagnose(r.path_params["run_id"])

async def run_action(r):
    rid, action = r.path_params["run_id"], r.path_params["action"]
    if action == "pause":
        return api.run_pause(rid, ACTOR)
    if action == "resume":
        return api.run_resume(rid, ACTOR)
    if action == "stop":
        return api.run_stop(rid, ACTOR, grace_seconds=10)
    if action == "retry":
        return api.run_retry(rid, ACTOR, BRAIN)
    if action == "adjust":
        return api.run_adjust(rid, ACTOR, (await _json(r))["changes"])
    raise api.BadRequest(f"unknown action '{action}'")

async def hpo_list(r):
    return api.hpo_list(50)

async def hpo_start(r):
    b = await _json(r)
    return api.hpo_start(b["plan"], b["metric"], int(b.get("n_trials", 10)), ACTOR, BRAIN,
                         b.get("direction") or None, b.get("search_space"), b.get("timeout_minutes"))

async def hpo_detail(r):
    return api.hpo_status(r.path_params["study_id"])

async def hpo_stop(r):
    return api.hpo_stop(r.path_params["study_id"], ACTOR)

async def models_list(r):
    return api.model_list()

async def model_register(r):
    b = await _json(r)
    return api.model_register(b["run_id"], b["name"], ACTOR, b.get("description", ""))

async def model_detail(r):
    return api.model_info(r.path_params["ref"])

async def model_predict(r):
    return api.model_predict(r.path_params["ref"], (await _json(r))["rows"])

async def model_export(r):
    res = api.model_export(r.path_params["ref"], r.query_params.get("format", "bundle"))
    return FileResponse(res["path"], filename=Path(res["path"]).name)


# ---------------------------------------------------------------- AI agent chat

def _ollama_models() -> list[str]:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    host = host if host.startswith("http") else "http://" + host
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=1) as resp:
            return sorted(m["name"] for m in json.loads(resp.read()).get("models", []))
    except (OSError, ValueError, KeyError):
        return []


async def agent_config(r):
    return {"default_brain": os.environ.get("AUTOFORGE_BRAIN") or DEFAULT_BRAIN,
            "keys": {"anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
                     "openai": bool(os.environ.get("OPENAI_API_KEY"))},
            "ollama_models": await asyncio.to_thread(_ollama_models)}


async def chats_list(r):
    return r.app.state.chats.list()


async def chat_create(r):
    body = await _json(r)
    spec = parse_brain(body.get("brain") or None, body.get("base_url") or None)
    provider = r.app.state.provider_factory(spec)
    return r.app.state.chats.create(spec, provider, int(body.get("max_steps") or 40)).summary()


async def chat_detail(r):
    return r.app.state.chats.get(r.path_params["chat_id"]).view(int(r.query_params.get("after", 0)))


async def chat_message(r):
    body = await _json(r)
    chat = r.app.state.chats.get(r.path_params["chat_id"])
    chat.send(str(body.get("text", "")), body.get("data_path") or None)
    return chat.summary()


async def chat_approval(r):
    body = await _json(r)
    chat = r.app.state.chats.get(r.path_params["chat_id"])
    chat.answer(bool(body.get("approved")), str(body.get("feedback") or ""))
    return chat.summary()


async def chat_stop(r):
    chat = r.app.state.chats.get(r.path_params["chat_id"])
    chat.stop()
    return chat.summary()


async def chat_close(r):
    r.app.state.chats.close(r.path_params["chat_id"])
    return {"closed": r.path_params["chat_id"]}


ROUTES = [
    ("GET", "/api/info", info), ("GET", "/api/hardware", hardware), ("GET", "/api/tasks", tasks),
    ("GET", "/api/data/profile", data_profile), ("GET", "/api/data/files", data_files),
    ("POST", "/api/data/upload", data_upload),
    ("GET", "/api/packs/search", packs_search), ("GET", "/api/packs", packs_installed),
    ("POST", "/api/packs/install", pack_install), ("GET", "/api/packs/{name}", pack_detail),
    ("POST", "/api/plans/validate", plan_validate),
    ("GET", "/api/runs", runs_list), ("POST", "/api/runs", run_start),
    ("GET", "/api/runs/{run_id}", run_detail), ("GET", "/api/runs/{run_id}/series", run_series),
    ("GET", "/api/runs/{run_id}/logs", run_logs), ("GET", "/api/runs/{run_id}/audit", run_audit),
    ("GET", "/api/runs/{run_id}/diagnose", run_diagnose), ("POST", "/api/runs/{run_id}/{action}", run_action),
    ("GET", "/api/hpo", hpo_list), ("POST", "/api/hpo", hpo_start),
    ("GET", "/api/hpo/{study_id}", hpo_detail), ("POST", "/api/hpo/{study_id}/stop", hpo_stop),
    ("GET", "/api/models", models_list), ("POST", "/api/models", model_register),
    ("GET", "/api/models/{ref}", model_detail), ("POST", "/api/models/{ref}/predict", model_predict),
    ("GET", "/api/models/{ref}/export", model_export),
    ("GET", "/api/agent/config", agent_config), ("GET", "/api/agent/chats", chats_list),
    ("POST", "/api/agent/chats", chat_create), ("GET", "/api/agent/chats/{chat_id}", chat_detail),
    ("POST", "/api/agent/chats/{chat_id}/messages", chat_message),
    ("POST", "/api/agent/chats/{chat_id}/approval", chat_approval),
    ("POST", "/api/agent/chats/{chat_id}/stop", chat_stop), ("DELETE", "/api/agent/chats/{chat_id}", chat_close),
]


def create_app(token: str, check_host: bool = True,
               provider_factory: Callable[[BrainSpec], Provider] = make_provider) -> Starlette:
    """`provider_factory` builds the agent's model client (tests pass a scripted fake)."""
    index_html = (STATIC / "index.html").read_text(encoding="utf-8")

    async def page(request: Request) -> Response:
        # The token is injected into the page; it is only reachable from the printed URL's origin.
        return Response(index_html.replace("__AUTOFORGE_TOKEN__", token), media_type="text/html",
                        headers={"Cache-Control": "no-store", "X-Frame-Options": "DENY",
                                 "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; "
                                                            "style-src 'self'; script-src 'self'; frame-ancestors 'none'"})

    class Guard(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            raw = request.headers.get("host", "")
            host = raw.split("]")[0] + "]" if raw.startswith("[") else raw.rsplit(":", 1)[0]
            if check_host and host not in ALLOWED_HOSTS:
                return _err("FORBIDDEN_HOST", f"host '{host}' not allowed", status=403)
            if request.url.path.startswith("/api/"):
                sent = request.headers.get("x-autoforge-token", "")
                if not hmac.compare_digest(sent, token):
                    return _err("UNAUTHORIZED", "missing or wrong GUI token", "open the URL printed by `autoforge gui`")
            return await call_next(request)

    routes = [Route(path, endpoint(fn), methods=[method]) for method, path, fn in ROUTES]
    routes += [Route("/", page), Mount("/static", StaticFiles(directory=STATIC), name="static")]
    chats = ChatManager()

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        yield
        chats.close_all()

    app = Starlette(routes=routes, middleware=[Middleware(Guard)], lifespan=lifespan)
    app.state.chats, app.state.provider_factory = chats, provider_factory
    return app
