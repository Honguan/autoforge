"""REST inference service: `autoforge serve`.

GET  /health
GET  /v1/models                     registered models (latest versions)
GET  /v1/models/{ref}               model card (ref = name, name:latest, name:v2)
POST /v1/models/{ref}/predict       {"rows": [{feature: value}, ...]} -> {"predictions": [...]}

Errors use the same envelope as the CLI/MCP: {"ok": false, "error": {code, message, hint}}.
Binds to 127.0.0.1 by default; set an API key to require `Authorization: Bearer <key>`.
"""

from __future__ import annotations

import hmac

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from autoforge import __version__
from autoforge.errors import AutoForgeError
from autoforge.registry import models as reg

STATUS = {"MODEL_NOT_FOUND": 404, "INVALID_INPUT": 422, "MODEL_NOT_SERVABLE": 409}


def _err(code: str, message: str, hint: str | None = None, status: int = 400, details=None) -> JSONResponse:
    body = {"ok": False, "data": details, "error": {"code": code, "message": message, "hint": hint}}
    return JSONResponse(body, status_code=status)


def create_app(allowed: list[str] | None = None, api_key: str | None = None) -> Starlette:
    """allowed: model names this server may expose (None = all registered models)."""

    def permitted(ref: str) -> bool:
        return allowed is None or reg.parse_ref(ref)[0] in allowed

    def guard(request: Request) -> JSONResponse | None:
        if api_key:
            sent = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(sent, api_key):
                return _err("UNAUTHORIZED", "missing or wrong API key", "send Authorization: Bearer <key>", 401)
        return None

    async def health(request: Request):
        return JSONResponse({"ok": True, "data": {"status": "up", "version": __version__}})

    async def list_models(request: Request):
        if (resp := guard(request)) is not None:
            return resp
        cards = [c.model_dump(mode="json") for c in reg.list_models() if permitted(c.name)]
        return JSONResponse({"ok": True, "data": cards})

    async def model_info(request: Request):
        if (resp := guard(request)) is not None:
            return resp
        ref = request.path_params["ref"]
        if not permitted(ref):
            return _err("MODEL_NOT_FOUND", f"model '{ref}' is not served here", status=404)
        try:
            return JSONResponse({"ok": True, "data": reg.get_card(ref).model_dump(mode="json")})
        except AutoForgeError as e:
            return _err(e.code, str(e), e.hint, STATUS.get(e.code, 400), e.details)

    async def predict(request: Request):
        if (resp := guard(request)) is not None:
            return resp
        ref = request.path_params["ref"]
        if not permitted(ref):
            return _err("MODEL_NOT_FOUND", f"model '{ref}' is not served here", status=404)
        try:
            body = await request.json()
        except ValueError:
            return _err("INVALID_INPUT", "body must be JSON: {\"rows\": [{...}]}", status=422)
        rows = body.get("rows") if isinstance(body, dict) else None
        try:
            return JSONResponse({"ok": True, "data": reg.predict(ref, rows)})
        except AutoForgeError as e:
            return _err(e.code, str(e), e.hint, STATUS.get(e.code, 400), e.details)

    return Starlette(routes=[
        Route("/health", health),
        Route("/v1/models", list_models),
        Route("/v1/models/{ref}", model_info),
        Route("/v1/models/{ref}/predict", predict, methods=["POST"]),
    ])
