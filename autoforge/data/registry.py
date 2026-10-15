"""Dataset sensitivity registry (~/.autoforge/datasets.json).

Marking a directory covers every file inside it. Unmarked data is 'internal'.
Only humans mark data (CLI); the MCP server exposes no way to change it, so a
cloud AI cannot downgrade a dataset to get access.
"""

from __future__ import annotations

import json
from pathlib import Path

from autoforge import config
from autoforge.schemas.core import Sensitivity

DEFAULT = Sensitivity.internal


def _file() -> Path:
    return config.home() / "datasets.json"


def _load() -> dict[str, str]:
    f = _file()
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}


def mark(path: str | Path, sensitivity: Sensitivity) -> str:
    key = str(Path(path).expanduser().resolve())
    data = _load() | {key: sensitivity.value}
    _file().parent.mkdir(parents=True, exist_ok=True)
    _file().write_text(json.dumps(dict(sorted(data.items())), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return key


def sensitivity_of(path: str | Path) -> Sensitivity:
    """The closest marked ancestor (or the path itself) decides; default 'internal'."""
    data = _load()
    p = Path(path).expanduser().resolve()
    for candidate in (p, *p.parents):
        if str(candidate) in data:
            return Sensitivity(data[str(candidate)])
    return DEFAULT


def all_marks() -> dict[str, str]:
    return _load()
