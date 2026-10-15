"""Paths and settings. Override the home directory with AUTOFORGE_HOME."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_INDEX_URL = (
    "https://raw.githubusercontent.com/Honguan/autoforge/packs-index/index.json"
)


def home() -> Path:
    return Path(os.environ.get("AUTOFORGE_HOME", Path.home() / ".autoforge")).expanduser()


def packs_dir() -> Path:
    return home() / "packs"


def lock_file() -> Path:
    return home() / "packs.lock.json"


def cache_dir() -> Path:
    return home() / "cache"


def index_url() -> str:
    """URL or local file path of the pack index."""
    return os.environ.get("AUTOFORGE_INDEX_URL", DEFAULT_INDEX_URL)
