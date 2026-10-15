"""MCP server that exposes trained models as tools: `autoforge serve-mcp`.

Each registered (servable) model becomes a tool `predict_<name>` whose input schema
is built from the model card, so any MCP client can call it without knowing
AutoForge. This server has no training or data tools: give it to AIs that should
only *use* models.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field, create_model

from autoforge import __version__
from autoforge.registry import models as reg
from autoforge.serve.mcp_server import _call

READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True)


def tool_name(model_name: str) -> str:
    return "predict_" + re.sub(r"[^a-zA-Z0-9_]", "_", model_name)


def _row_model(card: reg.ModelCard):
    fields: dict[str, Any] = {}
    for f in card.features:
        if f.dtype == "numeric":
            fields[f.name] = (float, Field(description="number"))
        else:
            known = f" Known values: {', '.join(f.known_values)}." if f.known_values else ""
            fields[f.name] = (str, Field(description="category." + known))
    return create_model(f"{tool_name(card.name)}_row", **fields)


def _description(card: reg.ModelCard) -> str:
    t = card.task
    parts = [card.description or f"Predict '{t.target}' ({t.task_type})."]
    if card.labels:
        parts.append(f"Possible predictions: {', '.join(card.labels)}.")
    if card.metrics:
        parts.append("Validation metrics: " + ", ".join(f"{k}={v:.3g}" for k, v in card.metrics.items()) + ".")
    parts.append(f"Model {card.ref}. Pass one object per example in `rows`; returns one prediction per row"
                 + (" with class probabilities." if card.labels else "."))
    return " ".join(parts)


def _make_tool(card: reg.ModelCard):
    ref = card.ref
    if card.features:
        Row = _row_model(card)

        def fn(rows):
            return _call(lambda: reg.predict(ref, [r.model_dump() for r in rows]))

        fn.__signature__ = inspect.Signature(
            [inspect.Parameter("rows", inspect.Parameter.KEYWORD_ONLY, annotation=list[Row])],
            return_annotation=dict[str, Any])
    else:
        def fn(rows):
            return _call(lambda: reg.predict(ref, rows))

        fn.__signature__ = inspect.Signature(
            [inspect.Parameter("rows", inspect.Parameter.KEYWORD_ONLY, annotation=list[dict[str, Any]])],
            return_annotation=dict[str, Any])
    fn.__name__ = tool_name(card.name)
    return fn


def build_model_server(names: list[str] | None = None) -> MCPServer:
    server = MCPServer("autoforge-models", version=__version__,
                       instructions="Trained models served by AutoForge. Call predict_<model> with rows of features.")
    cards = [c for c in reg.list_models() if c.servable and (names is None or c.name in names)]
    for card in cards:
        server.add_tool(_make_tool(card), name=tool_name(card.name), description=_description(card), annotations=READ)

    @server.tool(annotations=READ)
    def list_models() -> dict[str, Any]:
        """Models available here, with their input features, labels and metrics."""
        return _call(lambda: [
            {"tool": tool_name(c.name), "model": c.ref, "task": c.task.task_type, "target": c.task.target,
             "features": [f.model_dump() for f in c.features], "labels": c.labels, "metrics": c.metrics}
            for c in cards])

    return server


def main(names: list[str] | None = None) -> None:
    build_model_server(names).run()
