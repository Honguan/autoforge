"""Pack index: one index.json (normally on the packs-index branch)
that lets an AI find packs in a single structured call, without crawling GitHub."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from autoforge import config
from autoforge.packs.errors import IndexUnavailable, PackInvalid
from autoforge.packs.manifest import Pack
from autoforge.schemas.core import HardwareProfile


class IndexRecipe(BaseModel):
    id: str
    task_types: list[str]
    device: Literal["cpu", "cuda", "any"] = "any"
    min_vram_gb: float = 0.0
    good_for: str = ""


class IndexEntry(BaseModel):
    name: str
    version: str
    summary: str
    source: str = Field(description="'github:owner/repo[/subdir]' without '@ref'")
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$", description="pinned commit; verified on install")
    tasks: list[str]
    tags: list[str] = Field(default_factory=list)
    recipes: list[IndexRecipe] = Field(default_factory=list)

    @property
    def pinned_source(self) -> str:
        return f"{self.source}@{self.commit_sha}"


class PackIndex(BaseModel):
    schema_version: Literal[1] = 1
    packs: list[IndexEntry] = Field(default_factory=list)

    def get(self, name: str) -> IndexEntry | None:
        return next((p for p in self.packs if p.name == name), None)


class SearchHit(BaseModel):
    entry: IndexEntry
    runnable_recipes: list[str] = Field(description="recipes that fit the given hardware (all if no hardware)")
    score: int


def entry_from_pack(pack: Pack, source: str, commit_sha: str, tags: list[str] | None = None) -> IndexEntry:
    """Build an index entry from a validated pack (used by index maintainers)."""
    m = pack.manifest
    return IndexEntry(
        name=m.name,
        version=m.version,
        summary=m.summary,
        source=source,
        commit_sha=commit_sha,
        tasks=m.tasks,
        tags=tags or [],
        recipes=[
            IndexRecipe(
                id=r.id,
                task_types=r.task_types,
                device=r.requires.device,
                min_vram_gb=r.requires.min_vram_gb,
                good_for=r.good_for,
            )
            for r in pack.recipes
        ],
    )


def parse_index(text: str) -> PackIndex:
    try:
        index = PackIndex.model_validate_json(text)
    except ValidationError as e:
        raise PackInvalid([f"index: {'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()])
    names = [p.name for p in index.packs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise PackInvalid([f"index: duplicate pack names: {dupes}"])
    return index


def load_index(refresh: bool = True) -> tuple[PackIndex, list[str]]:
    """Load the index from AUTOFORGE_INDEX_URL (URL or file); fall back to the cache."""
    notes: list[str] = []
    src = config.index_url()
    cache = config.cache_dir() / "index.json"
    text = None
    if Path(src).is_file():
        text = Path(src).read_text(encoding="utf-8")
    elif refresh and src.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(src, timeout=15) as resp:
                text = resp.read().decode("utf-8")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
        except OSError as e:
            notes.append(f"could not fetch index ({e}); using cached copy")
    if text is None:
        if not cache.is_file():
            raise IndexUnavailable(f"pack index unavailable: {src}")
        text = cache.read_text(encoding="utf-8")
    return parse_index(text), notes


def _recipe_fits(r: IndexRecipe, hw: HardwareProfile) -> bool:
    if r.device == "cpu":
        return True
    has_cuda = hw.best_device == "cuda"
    if r.device == "cuda":
        return has_cuda and hw.max_vram_gb >= r.min_vram_gb
    return r.min_vram_gb == 0 or (has_cuda and hw.max_vram_gb >= r.min_vram_gb)


def search(
    index: PackIndex,
    task: str | None = None,
    query: str | None = None,
    hardware: HardwareProfile | None = None,
) -> list[SearchHit]:
    terms = [t.lower() for t in (query or "").split()]
    hits = []
    for e in index.packs:
        if task and task not in e.tasks:
            continue
        haystack = " ".join([e.name, e.summary, *e.tags, *e.tasks, *(r.good_for for r in e.recipes)]).lower()
        score = sum(t in haystack for t in terms)
        if terms and score == 0:
            continue
        recipes = [r for r in e.recipes if task is None or task in r.task_types]
        if hardware is not None:
            recipes = [r for r in recipes if _recipe_fits(r, hardware)]
            if not recipes:
                continue
        hits.append(SearchHit(entry=e, runnable_recipes=[r.id for r in recipes], score=score))
    return sorted(hits, key=lambda h: (-h.score, h.entry.name))


def dump_index(index: PackIndex) -> str:
    return json.dumps(index.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
