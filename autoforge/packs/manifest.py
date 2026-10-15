"""Pack format: pack.yaml + KNOWLEDGE.md + recipes/*.yaml + trainer code.

Validation never imports or executes pack code; entrypoints are checked by
parsing the source with `ast`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from autoforge.packs.errors import PackInvalid

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+([-+][0-9A-Za-z.-]+)?$")


class Requirements(BaseModel):
    device: Literal["cpu", "cuda", "any"] = "any"
    min_vram_gb: float = 0.0
    min_ram_gb: float = 0.0


class RecipeSpec(BaseModel):
    """One training method inside a pack (recipes/<id>.yaml)."""

    id: str
    name: str
    task_types: list[str] = Field(min_length=1)
    entrypoint: str = Field(description="'trainers/file.py:function', relative to the pack root")
    requires: Requirements = Field(default_factory=Requirements)
    good_for: str = ""
    avoid_when: str = ""
    defaults: dict[str, Any] = Field(default_factory=dict)
    search_space: dict[str, Any] = Field(
        default_factory=dict, description="hyperparameter -> {type, low, high} or {choices}"
    )
    predictor: str | None = Field(
        default=None,
        description="module 'trainers/file.py' defining load_model(model_dir) and predict(model, rows) for serving",
    )
    exporters: dict[str, str] = Field(
        default_factory=dict, description="format -> 'file.py:function', e.g. {onnx: trainers/x.py:export_onnx}"
    )

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not re.fullmatch(r"^[a-z0-9_]+$", v):
            raise ValueError("recipe id must be lowercase letters, digits or '_'")
        return v


class PackManifest(BaseModel):
    """pack.yaml"""

    name: str
    version: str
    summary: str
    tasks: list[str] = Field(min_length=1)
    recipes: list[str] = Field(min_length=1, description="recipe ids; each has recipes/<id>.yaml")
    dependencies: list[str] = Field(default_factory=list, description="pip requirement strings")
    authors: list[str] = Field(default_factory=list)
    license: str | None = None
    min_autoforge: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not NAME_RE.fullmatch(v):
            raise ValueError("name must be lowercase kebab-case, e.g. 'text-classification'")
        return v

    @field_validator("version")
    @classmethod
    def _version(cls, v: str) -> str:
        if not SEMVER_RE.fullmatch(v):
            raise ValueError("version must be semver, e.g. '1.0.0'")
        return v


class Pack(BaseModel):
    root: Path
    manifest: PackManifest
    recipes: list[RecipeSpec]

    def knowledge(self) -> str:
        return (self.root / "KNOWLEDGE.md").read_text(encoding="utf-8")

    def recipe(self, recipe_id: str) -> RecipeSpec | None:
        return next((r for r in self.recipes if r.id == recipe_id), None)


def _read_yaml(path: Path, problems: list[str]) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        problems.append(f"missing file: {path.name}")
    except yaml.YAMLError as e:
        problems.append(f"{path.name}: invalid YAML: {e}")
    return None


def _fmt(e: ValidationError, where: str) -> list[str]:
    return [f"{where}: {'.'.join(map(str, err['loc'])) or '<root>'}: {err['msg']}" for err in e.errors()]


def _check_functions(root: Path, rel: str, funcs: list[str], where: str, label: str) -> list[str]:
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        return [f"{where}: {label} escapes the pack directory"]
    if not path.is_file():
        return [f"{where}: {label} file not found: {rel}"]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return [f"{where}: {rel} has a syntax error: {e}"]
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    return [f"{where}: function '{f}' not defined at top level of {rel}" for f in funcs if f not in defined]


def _check_entrypoint(root: Path, entrypoint: str, where: str, label: str = "entrypoint") -> list[str]:
    if ":" not in entrypoint:
        return [f"{where}: {label} must look like 'trainers/file.py:function'"]
    rel, func = entrypoint.rsplit(":", 1)
    return _check_functions(root, rel, [func], where, label)


def collect_problems(root: Path) -> tuple[Pack | None, list[str]]:
    """Validate a pack directory. Returns (pack, problems); pack is None if invalid."""
    root = Path(root)
    problems: list[str] = []
    if not root.is_dir():
        return None, [f"not a directory: {root}"]

    raw = _read_yaml(root / "pack.yaml", problems)
    manifest = None
    if raw is not None:
        try:
            manifest = PackManifest.model_validate(raw)
        except ValidationError as e:
            problems += _fmt(e, "pack.yaml")

    knowledge = root / "KNOWLEDGE.md"
    if not knowledge.is_file():
        problems.append("missing file: KNOWLEDGE.md")
    elif len(knowledge.read_text(encoding="utf-8").strip()) < 50:
        problems.append("KNOWLEDGE.md is too short to teach an AI anything (< 50 chars)")

    recipes: list[RecipeSpec] = []
    if manifest:
        if len(set(manifest.recipes)) != len(manifest.recipes):
            problems.append("pack.yaml: duplicate recipe ids")
        for rid in manifest.recipes:
            where = f"recipes/{rid}.yaml"
            data = _read_yaml(root / "recipes" / f"{rid}.yaml", problems)
            if data is None:
                continue
            try:
                recipe = RecipeSpec.model_validate(data)
            except ValidationError as e:
                problems += _fmt(e, where)
                continue
            if recipe.id != rid:
                problems.append(f"{where}: id '{recipe.id}' does not match file name")
            unknown = set(recipe.task_types) - set(manifest.tasks)
            if unknown:
                problems.append(f"{where}: task_types not declared in pack.yaml tasks: {sorted(unknown)}")
            if recipe.requires.min_vram_gb > 0 and recipe.requires.device == "cpu":
                problems.append(f"{where}: device 'cpu' cannot require VRAM")
            problems += _check_entrypoint(root, recipe.entrypoint, where)
            if recipe.predictor:
                problems += _check_functions(root, recipe.predictor, ["load_model", "predict"], where, "predictor")
            for fmt, ep in recipe.exporters.items():
                problems += _check_entrypoint(root, ep, where, f"exporter '{fmt}'")
            recipes.append(recipe)

    if problems or manifest is None:
        return None, problems
    return Pack(root=root, manifest=manifest, recipes=recipes), []


def load_pack(root: Path) -> Pack:
    pack, problems = collect_problems(root)
    if pack is None:
        raise PackInvalid(problems)
    return pack
