"""Model registry (P6): turn a successful run into a versioned, self-contained model.

models/<name>/v<N>/
  card.json   ModelCard: source run, task, metrics, input features, labels, sensitivity
  model/      the run's output directory (weights, joblib, ...)
  code/       the run's code snapshot (pack or custom), including the predictor module

A registered model never depends on the currently installed pack version: it is
served with the exact code that trained it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from autoforge import config
from autoforge.data.profile import profile
from autoforge.data.registry import sensitivity_of
from autoforge.engine import runner
from autoforge.engine.store import RunStore, now_iso
from autoforge.errors import AutoForgeError
from autoforge.policy.privacy import Access
from autoforge.schemas.core import RunStatus, Sensitivity, TaskSpec

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


class ModelError(AutoForgeError):
    code = "MODEL_ERROR"


class ModelNotFound(ModelError):
    code = "MODEL_NOT_FOUND"
    hint = "Use `autoforge model list` (MCP: model_list) to see registered models."


class ModelNotServable(ModelError):
    code = "MODEL_NOT_SERVABLE"
    hint = "The recipe (or custom trainer) has no predictor: it must define load_model(model_dir) and predict(model, rows)."


class InvalidInput(ModelError):
    code = "INVALID_INPUT"
    hint = "Send rows as a list of objects with every feature listed in the model card (model_info)."


class Feature(BaseModel):
    name: str
    dtype: Literal["numeric", "categorical", "text", "image", "audio", "any"]
    description: str = ""
    required: bool = True
    known_values: list[str] | None = Field(default=None, description="seen in training (low-cardinality columns only)")


class ModelCard(BaseModel):
    name: str
    version: int
    created_at: str
    actor: str
    description: str = ""
    run_id: str
    task: TaskSpec
    pack: str | None = None
    pack_version: str | None = None
    pack_source: str | None = None
    recipe: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    features: list[Feature] = Field(default_factory=list)
    labels: list[str] | None = None
    data_sensitivity: Sensitivity
    servable: bool
    predictor: str | None = Field(default=None, description="predictor module, relative to the version directory")
    exporters: dict[str, str] = Field(default_factory=dict, description="format -> 'relative/file.py:function'")

    @property
    def ref(self) -> str:
        return f"{self.name}:v{self.version}"


def models_dir() -> Path:
    return config.home() / "models"


def _version_dir(name: str, version: int) -> Path:
    return models_dir() / name / f"v{version}"


def versions(name: str) -> list[int]:
    d = models_dir() / name
    if not d.is_dir():
        return []
    return sorted(int(p.name[1:]) for p in d.iterdir() if p.name[1:].isdigit() and (p / "card.json").is_file())


def parse_ref(ref: str) -> tuple[str, int | None]:
    """'churn' -> latest, 'churn:v2' or 'churn:2' -> version 2."""
    name, _, ver = ref.partition(":")
    if not ver or ver == "latest":
        return name, None
    ver = ver.lstrip("v")
    if not ver.isdigit():
        raise ModelNotFound(f"bad model reference '{ref}'", hint="Use name, name:latest or name:v<N>.")
    return name, int(ver)


def get_card(ref: str) -> ModelCard:
    name, version = parse_ref(ref)
    vs = versions(name)
    if not vs:
        raise ModelNotFound(f"no model named '{name}'")
    version = version or vs[-1]
    if version not in vs:
        raise ModelNotFound(f"model '{name}' has no version v{version}; versions: {vs}")
    return ModelCard.model_validate_json((_version_dir(name, version) / "card.json").read_text(encoding="utf-8"))


def list_models() -> list[ModelCard]:
    d = models_dir()
    if not d.is_dir():
        return []
    return [get_card(f"{p.name}:v{versions(p.name)[-1]}") for p in sorted(d.iterdir()) if versions(p.name)]


# ---------- register ----------


def _features(data_path: str, target: str | None, sensitivity: Sensitivity,
              classification: bool) -> tuple[list[Feature], list[str] | None]:
    p = Path(data_path)
    if not (p.is_file() and p.suffix.lower() == ".csv"):
        return [], None
    prof = profile(p, Access.summary)
    hide = sensitivity is Sensitivity.sensitive  # don't copy category values of sensitive data into the card
    feats, labels = [], None
    for col, info in prof["columns"].items():
        values = list(info["top_values"]) if "top_values" in info and not hide else None
        if col == target:
            labels = sorted(values) if classification and values else None
            continue
        feats.append(Feature(name=col, dtype=info["dtype"], known_values=values if info["dtype"] == "categorical" else None))
    return feats, labels


def register(run_id: str, name: str, actor: str = "human", description: str = "") -> ModelCard:
    if not NAME_RE.fullmatch(name):
        raise ModelError(f"bad model name '{name}'", hint="lowercase letters, digits, '-' or '_' (max 63 chars)")
    store = RunStore.open(run_id)
    view, rec = runner.status(store), store.record()
    if view.status is not RunStatus.succeeded:
        raise ModelError(f"run {run_id} is {view.status.value}; only succeeded runs can be registered")
    out_dir = store.dir / "output"
    if not out_dir.is_dir() or not any(out_dir.iterdir()):
        raise ModelError(f"run {run_id} produced no files in its output directory")

    version = (versions(name)[-1] if versions(name) else 0) + 1
    vdir = _version_dir(name, version)
    vdir.mkdir(parents=True)
    shutil.copytree(out_dir, vdir / "model")

    predictor, exporters = None, {}
    code_src = store.dir / "pack" if (store.dir / "pack").is_dir() else store.dir / "code"
    if code_src.is_dir():
        shutil.copytree(code_src, vdir / "code", ignore=shutil.ignore_patterns("__pycache__"))
        if rec.predictor:
            predictor = str(Path("code") / Path(rec.predictor).relative_to(code_src))
        for fmt, ep in rec.exporters.items():
            file, func = ep.rsplit(":", 1)
            exporters[fmt] = f"{Path('code') / Path(file).relative_to(code_src)}:{func}"

    sens = sensitivity_of(rec.plan.data_path)
    io_file = out_dir / "io.json"
    if io_file.is_file():  # the trainer described its own inputs (text, images, prompts, history...)
        io = json.loads(io_file.read_text(encoding="utf-8"))
        feats = [Feature.model_validate(f) for f in io.get("features", [])]
        labels = [str(x) for x in io["labels"]] if io.get("labels") else None
        if sens is Sensitivity.sensitive:
            feats = [f.model_copy(update={"known_values": None}) for f in feats]
            labels = None
    else:
        feats, labels = _features(rec.plan.data_path, rec.plan.task.target, sens,
                                  "classification" in rec.plan.task.task_type)
    card = ModelCard(
        name=name, version=version, created_at=now_iso(), actor=actor, description=description,
        run_id=run_id, task=rec.plan.task, pack=rec.plan.pack, pack_version=rec.pack_version,
        pack_source=rec.pack_source, recipe=rec.plan.recipe, hyperparameters=rec.plan.hyperparameters,
        metrics=view.metrics, features=feats, labels=labels, data_sensitivity=sens,
        servable=predictor is not None, predictor=predictor, exporters=exporters,
    )
    (vdir / "card.json").write_text(card.model_dump_json(indent=2), encoding="utf-8")
    store.audit(actor, "register_model", model=card.ref)
    return card


def remove(ref: str) -> list[str]:
    name, version = parse_ref(ref)
    targets = [version] if version else versions(name)
    if not targets or any(v not in versions(name) for v in targets):
        raise ModelNotFound(f"no such model '{ref}'")
    for v in targets:
        shutil.rmtree(_version_dir(name, v))
    _CACHE.clear()
    if not versions(name):
        shutil.rmtree(models_dir() / name, ignore_errors=True)
    return [f"{name}:v{v}" for v in targets]


# ---------- predict ----------

_CACHE: dict[str, tuple[ModelCard, Any, Any]] = {}


def _load_module(path: Path, key: str):
    spec = importlib.util.spec_from_file_location(f"autoforge_predictor_{key}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def load(ref: str) -> tuple[ModelCard, Any, Any]:
    card = get_card(ref)
    if card.ref in _CACHE:
        return _CACHE[card.ref]
    if not card.servable:
        raise ModelNotServable(f"model {card.ref} cannot be served")
    vdir = _version_dir(card.name, card.version)
    module = _load_module(vdir / card.predictor, re.sub(r"\W", "_", card.ref))
    model = module.load_model(vdir / "model")
    _CACHE[card.ref] = (card, module, model)
    return _CACHE[card.ref]


def validate_rows(card: ModelCard, rows: Any) -> list[str]:
    """Raise InvalidInput on missing features or wrong types; return warnings."""
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise InvalidInput("rows must be a non-empty list of objects")
    if len(rows) > 10_000:
        raise InvalidInput("at most 10000 rows per request")
    if not card.features:
        return []
    names = {f.name for f in card.features}
    problems, warnings = [], []
    for i, row in enumerate(rows):
        missing = [f.name for f in card.features if f.required and f.name not in row]
        if missing:
            problems.append(f"row {i}: missing {missing}")
        for f in card.features:
            v = row.get(f.name)
            if f.dtype == "numeric" and v is not None and not isinstance(v, bool):
                try:
                    float(v)
                except (TypeError, ValueError):
                    problems.append(f"row {i}: '{f.name}' must be a number, got {v!r}")
            if f.dtype == "categorical" and f.known_values and v is not None and str(v) not in f.known_values:
                warnings.append(f"row {i}: '{f.name}'={v!r} was not seen in training")
            if f.dtype in ("text", "image", "audio") and v is not None and not isinstance(v, str):
                problems.append(f"row {i}: '{f.name}' must be a string")
        extra = sorted(set(row) - names)
        if extra:
            warnings.append(f"row {i}: ignored unknown fields {extra}")
    if problems:
        raise InvalidInput("; ".join(problems[:20]), details={"problems": problems[:100],
                                                              "features": [f.model_dump() for f in card.features]})
    return warnings[:50]


def predict(ref: str, rows: list[dict]) -> dict:
    card, module, model = load(ref)
    warnings = validate_rows(card, rows)
    try:
        preds = module.predict(model, rows)
    except Exception as e:
        raise ModelError(f"prediction failed: {type(e).__name__}: {e}")
    return {"model": card.ref, "predictions": preds, "warnings": warnings}


# ---------- export / import ----------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(ref: str, fmt: str, out: Path) -> dict:
    card = get_card(ref)
    vdir = _version_dir(card.name, card.version)
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "bundle":
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(vdir.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(vdir).as_posix())
    elif fmt in card.exporters:
        file, func = card.exporters[fmt].rsplit(":", 1)
        module = _load_module(vdir / file, re.sub(r"\W", "_", card.ref) + "_export")
        features = {f.name: f.dtype for f in card.features}
        try:
            getattr(module, func)(vdir / "model", out, features)
        except ImportError as e:
            raise ModelError(f"export to {fmt} needs an extra package: {e}",
                             hint="pip install skl2onnx" if fmt == "onnx" else None)
        if out.is_dir():  # e.g. merged LLM weights: ship one archive
            archive = Path(shutil.make_archive(str(out), "zip", out))
            shutil.rmtree(out)
            out = archive
    else:
        raise ModelError(f"model {card.ref} cannot be exported as '{fmt}'",
                         hint=f"available: {['bundle', *card.exporters]}")
    return {"model": card.ref, "format": fmt, "path": str(out), "bytes": out.stat().st_size, "sha256": _sha256(out)}


def import_bundle(path: Path, name: str | None = None, actor: str = "human") -> ModelCard:
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if member.startswith("/") or ".." in Path(member).parts:
                raise ModelError(f"unsafe path in bundle: {member}")
        try:
            card = ModelCard.model_validate_json(z.read("card.json"))
        except KeyError:
            raise ModelError("bundle has no card.json")
        name = name or card.name
        if not NAME_RE.fullmatch(name):
            raise ModelError(f"bad model name '{name}'")
        version = (versions(name)[-1] if versions(name) else 0) + 1
        vdir = _version_dir(name, version)
        vdir.mkdir(parents=True)
        z.extractall(vdir)
    card = card.model_copy(update={"name": name, "version": version, "created_at": now_iso(), "actor": actor})
    (vdir / "card.json").write_text(card.model_dump_json(indent=2), encoding="utf-8")
    return card
