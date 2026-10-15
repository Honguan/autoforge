"""Deterministic plan checks (no AI involved). A wrong AI plan stops here."""

from __future__ import annotations

import ast
import csv
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from autoforge.errors import AutoForgeError
from autoforge.hardware.detect import detect_hardware
from autoforge.packs import installer
from autoforge.packs.errors import PackError
from autoforge.schemas.core import HardwareProfile, TrainingPlan


class PlanInvalid(AutoForgeError):
    code = "PLAN_INVALID"
    hint = "Fix every item in details.problems, then run `autoforge plan validate` again."


class PlanCheck(BaseModel):
    ok: bool
    problems: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    device: str | None = None
    entrypoint: str | None = Field(default=None, description="absolute 'file.py:function'")
    hyperparameters: dict = Field(default_factory=dict, description="recipe defaults merged with the plan")
    pack_version: str | None = None
    pack_source: str | None = None


def load_plan_file(path: Path) -> TrainingPlan:
    """Load a YAML/JSON plan; relative paths are resolved against the plan file's directory."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PlanInvalid(f"plan file not found: {path}")
    except yaml.YAMLError as e:
        raise PlanInvalid(f"plan file is not valid YAML/JSON: {e}")
    return parse_plan(raw, path.resolve().parent)


def parse_plan(raw: object, base: Path) -> TrainingPlan:
    """Validate a plan dict; relative paths are resolved against `base`."""
    try:
        plan = TrainingPlan.model_validate(raw)
    except ValidationError as e:
        problems = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
        raise PlanInvalid("plan does not match the TrainingPlan schema", details={"problems": problems})
    plan.data_path = str((base / plan.data_path).resolve())
    if plan.custom_code_path:
        file, _, func = plan.custom_code_path.partition(":")
        plan.custom_code_path = str((base / file).resolve()) + (f":{func}" if func else "")
    return plan


def _function_defined(file: Path, func: str) -> bool:
    try:
        tree = ast.parse(file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    return any(isinstance(n, ast.FunctionDef) and n.name == func for n in tree.body)


def _csv_columns(path: Path) -> list[str] | None:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return next(csv.reader(f), [])
    except (OSError, UnicodeDecodeError):
        return None


def check_plan(plan: TrainingPlan, hardware: HardwareProfile | None = None) -> PlanCheck:
    hw = hardware or detect_hardware()
    problems: list[str] = []
    warnings: list[str] = []
    out = PlanCheck(ok=False)
    has_cuda = hw.best_device == "cuda"

    # data
    data = Path(plan.data_path)
    if not data.exists():
        problems.append(f"data_path not found: {plan.data_path}")
    elif data.suffix.lower() == ".csv" and plan.task.target:
        cols = _csv_columns(data)
        if cols is not None and plan.task.target not in cols:
            problems.append(f"target column '{plan.task.target}' not in CSV columns {cols}")

    # method: exactly one of pack+recipe or custom code
    required_device = "any"
    if plan.custom_code_path and plan.pack:
        problems.append("set either pack+recipe or custom_code_path, not both")
    elif plan.custom_code_path:
        file, _, func = plan.custom_code_path.partition(":")
        func = func or "train"
        if not Path(file).is_file():
            problems.append(f"custom code not found: {file}")
        elif not _function_defined(Path(file), func):
            problems.append(f"function '{func}' not defined at top level of {file}")
        else:
            out.entrypoint = f"{Path(file).resolve()}:{func}"
        out.hyperparameters = dict(plan.hyperparameters)
        warnings.append("custom code runs with full permissions; it is versioned in the run's git workspace")
    elif plan.pack:
        try:
            rec, pack = installer.get_installed(plan.pack)
        except PackError as e:
            problems.append(str(e))
        else:
            out.pack_version, out.pack_source = rec.version, rec.source
            recipe = pack.recipe(plan.recipe or "")
            if recipe is None:
                problems.append(
                    f"recipe '{plan.recipe}' not in pack '{plan.pack}'; available: {[r.id for r in pack.recipes]}"
                )
            else:
                if plan.task.task_type not in recipe.task_types:
                    problems.append(
                        f"recipe '{recipe.id}' does not support task '{plan.task.task_type}' "
                        f"(supports {recipe.task_types})"
                    )
                req = recipe.requires
                required_device = req.device
                if req.device == "cuda" and not has_cuda:
                    problems.append(f"recipe '{recipe.id}' needs an NVIDIA GPU with CUDA; none usable")
                if req.min_vram_gb and has_cuda and hw.max_vram_gb < req.min_vram_gb:
                    problems.append(f"recipe needs {req.min_vram_gb} GB VRAM; largest GPU has {hw.max_vram_gb} GB")
                if req.min_ram_gb and hw.ram_available_gb < req.min_ram_gb:
                    warnings.append(f"recipe suggests {req.min_ram_gb} GB free RAM; {hw.ram_available_gb} GB available")
                unknown = set(plan.hyperparameters) - set(recipe.defaults) - set(recipe.search_space)
                if unknown:
                    warnings.append(f"hyperparameters not declared by the recipe: {sorted(unknown)}")
                file, func = recipe.entrypoint.rsplit(":", 1)
                out.entrypoint = f"{(pack.root / file).resolve()}:{func}"
                out.hyperparameters = recipe.defaults | plan.hyperparameters
            missing = installer.missing_dependencies(pack)
            if missing:
                problems.append(f"missing Python packages: {missing} (pip install them)")
    else:
        problems.append("plan needs pack+recipe or custom_code_path")

    # device
    if plan.device == "cuda" and not has_cuda:
        problems.append("plan asks for device 'cuda' but no usable NVIDIA GPU/CUDA was found")
    if plan.device == "cuda" and required_device == "cpu":
        problems.append("recipe is CPU-only but plan asks for 'cuda'")
    if required_device in ("cpu", "cuda"):
        out.device = required_device
    elif plan.device != "auto":
        out.device = plan.device
    else:
        out.device = hw.best_device

    out.problems, out.warnings = problems, warnings
    out.ok = not problems
    return out


def require_valid(plan: TrainingPlan, hardware: HardwareProfile | None = None) -> PlanCheck:
    check = check_plan(plan, hardware)
    if not check.ok:
        raise PlanInvalid(f"{len(check.problems)} problem(s) in plan", details=check.model_dump())
    return check
