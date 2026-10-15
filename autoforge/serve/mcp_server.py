"""MCP server: lets any MCP client (Claude Desktop, Claude Code, ...) drive AutoForge.

Start with `autoforge mcp` (stdio). Every tool returns the Envelope shape:
{"ok": true, "data": ...} or {"ok": false, "error": {"code", "message", "hint"}, "data": details}.
"""

from __future__ import annotations

from typing import Any, Callable

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from autoforge import __version__, api
from autoforge.errors import AutoForgeError
from autoforge.policy.privacy import brain_from_actor
from autoforge.schemas.core import SCHEMAS, BrainKind, Envelope, ErrorInfo

INSTRUCTIONS = """\
AutoForge trains AI models on the user's own machine. Typical workflow:
1. hardware_profile — learn CPU/GPU before choosing a method.
2. data_profile(path) — inspect the dataset (statistics; raw rows only for public data).
3. pack_search(query, task) → pack_install(name) → pack_info(name): READ `knowledge` before planning.
4. Write a TrainingPlan (schema_get('TrainingPlan')): task, absolute data_path, pack + recipe, hyperparameters.
   Or write your own trainer with trainer_write and set custom_code_path instead of pack/recipe.
5. plan_validate(plan) and fix every `problems` item, then run_start(plan) → run_id.
6. run_wait(run_id) / run_status / run_metrics while it trains; run_logs on failure.
7. run_pause / run_resume / run_stop / run_retry to control it. The user can control the same run.
Every result is {"ok": bool, "data", "error": {code, message, hint}}. Branch on error.code and follow hint.
PRIVACY_DENIED means the dataset is sensitive: never retry or work around it; tell the user.
Always use absolute paths."""

READ = ToolAnnotations(readOnlyHint=True)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True)


def _call(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        return Envelope(ok=True, data=fn()).model_dump(mode="json")
    except AutoForgeError as e:
        err = ErrorInfo(code=e.code, message=str(e), hint=e.hint)
        return Envelope(ok=False, error=err, data=e.details).model_dump(mode="json")
    except Exception as e:  # never crash the session; report it
        err = ErrorInfo(code="INTERNAL_ERROR", message=f"{type(e).__name__}: {e}")
        return Envelope(ok=False, error=err).model_dump(mode="json")


def build_server(actor: str = "ai:mcp", log_level: str = "INFO") -> MCPServer:
    """`actor` is recorded in audit logs. 'ai:local:<model>' marks a local-model client,
    which is the only AI allowed to use sensitive data."""
    brain = brain_from_actor(actor)
    if brain is BrainKind.api:
        brain = BrainKind.subscription
    server = MCPServer("autoforge", version=__version__, instructions=INSTRUCTIONS, log_level=log_level)
    tool = server.tool

    # ---- environment ----
    @tool(annotations=READ)
    def hardware_profile() -> dict[str, Any]:
        """CPU, RAM, disk, NVIDIA GPUs (VRAM, compute capability), CUDA/torch status and best_device."""
        return _call(api.hardware)

    @tool(annotations=READ)
    def schema_get(name: str) -> dict[str, Any]:
        """JSON Schema of a data structure, e.g. TrainingPlan, TaskSpec, RecipeSpec, PackManifest, RunView."""
        from autoforge.cli.main import ALL_SCHEMAS

        def get():
            if name not in ALL_SCHEMAS:
                raise api.BadRequest(f"unknown schema '{name}'", hint=f"choose from {sorted(ALL_SCHEMAS)}")
            return ALL_SCHEMAS[name].model_json_schema()

        return _call(get)

    # ---- packs ----
    @tool(annotations=READ)
    def pack_search(query: str | None = None, task: str | None = None, fit_hardware: bool = True) -> dict[str, Any]:
        """Search the method-pack index by keywords and/or exact task type (e.g. tabular-classification).
        With fit_hardware, only recipes this machine can run are listed (runnable_recipes)."""
        return _call(lambda: api.pack_search(query, task, fit_hardware))

    @tool(annotations=WRITE)
    def pack_install(source: str, force: bool = False) -> dict[str, Any]:
        """Install a pack: an index name (pinned commit, verified), 'github:owner/repo[/subdir][@ref]', or a local path."""
        return _call(lambda: api.pack_install(source, force))

    @tool(annotations=READ)
    def pack_info(name: str, include_knowledge: bool = True) -> dict[str, Any]:
        """An installed pack's recipes (requirements, defaults, search space), missing pip packages and
        KNOWLEDGE.md (how to choose and tune recipes). Read it before planning."""
        return _call(lambda: api.pack_info(name, include_knowledge))

    @tool(annotations=READ)
    def pack_list() -> dict[str, Any]:
        """Installed packs with version and pinned source."""
        return _call(api.pack_list)

    # ---- data ----
    @tool(annotations=READ)
    def data_profile(path: str) -> dict[str, Any]:
        """Profile a dataset file or directory: rows, columns, dtypes, missing values, value counts.
        Sample rows are included only for data marked public. Sensitive data is refused."""
        return _call(lambda: api.data_profile(path, brain))

    # ---- custom code ----
    @tool(annotations=WRITE)
    def trainer_write(filename: str, source: str) -> dict[str, Any]:
        """Save your own trainer (Python) to the AutoForge workspace. Contract:
        def train(ctx) -> {"metrics": {...}, "model_path": str}; ctx has data_path, target, task_type,
        hyperparameters, device, seed, output_dir, checkpoint_dir, log(msg), log_metric(name, value, step),
        check_stop(), latest_checkpoint(). Returns the path to use as custom_code_path in a plan."""
        return _call(lambda: api.trainer_write(filename, source))

    # ---- plans / runs ----
    @tool(annotations=READ)
    def plan_validate(plan: dict[str, Any]) -> dict[str, Any]:
        """Check a TrainingPlan without running it: pack/recipe/task match, data and target column,
        GPU/VRAM, missing packages, privacy. Returns problems, warnings, device and merged hyperparameters."""
        return _call(lambda: api.plan_validate(plan, actor, brain))

    @tool(annotations=WRITE)
    def run_start(plan: dict[str, Any]) -> dict[str, Any]:
        """Validate and start training in the background. Returns immediately with run_id and status."""
        return _call(lambda: api.run_start(plan, actor, brain))

    @tool(annotations=READ)
    def run_status(run_id: str) -> dict[str, Any]:
        """Status (pending/running/paused/succeeded/failed/stopped), final metrics, latest metrics, model_path, error."""
        return _call(lambda: api.run_status(run_id))

    @tool(annotations=READ)
    def run_wait(run_id: str, timeout_seconds: float = 30) -> dict[str, Any]:
        """Wait up to timeout_seconds (max 120) for a run to finish, then return its status."""
        return _call(lambda: api.run_wait(run_id, timeout_seconds))

    @tool(annotations=READ)
    def run_list(limit: int = 20) -> dict[str, Any]:
        """Most recent runs, newest first."""
        return _call(lambda: api.run_list(limit))

    @tool(annotations=READ)
    def run_metrics(run_id: str, last: int = 5) -> dict[str, Any]:
        """Per metric: last, min, max, count and the most recent values (with steps)."""
        return _call(lambda: api.run_metrics(run_id, last))

    @tool(annotations=READ)
    def run_logs(run_id: str, tail: int = 50) -> dict[str, Any]:
        """Last lines (max 500) of the run's stdout/stderr. Use it to diagnose failures."""
        return _call(lambda: api.run_logs(run_id, tail))

    @tool(annotations=READ)
    def run_audit(run_id: str) -> dict[str, Any]:
        """Every action on a run and who took it (human or ai:...)."""
        return _call(lambda: api.run_audit(run_id))

    @tool(annotations=WRITE)
    def run_pause(run_id: str) -> dict[str, Any]:
        """Suspend a running run."""
        return _call(lambda: api.run_pause(run_id, actor))

    @tool(annotations=WRITE)
    def run_resume(run_id: str) -> dict[str, Any]:
        """Continue a paused run."""
        return _call(lambda: api.run_resume(run_id, actor))

    @tool(annotations=DESTRUCTIVE)
    def run_stop(run_id: str, grace_seconds: float = 10.0) -> dict[str, Any]:
        """Stop a run: ask it to stop cleanly, terminate after grace_seconds."""
        return _call(lambda: api.run_stop(run_id, actor, grace_seconds))

    @tool(annotations=WRITE)
    def run_retry(run_id: str) -> dict[str, Any]:
        """Start a new run with the same plan, resuming from the old run's checkpoints if any."""
        return _call(lambda: api.run_retry(run_id, actor, brain))

    return server


def main(actor: str = "ai:mcp") -> None:
    build_server(actor).run()  # stdio


if __name__ == "__main__":
    main()
