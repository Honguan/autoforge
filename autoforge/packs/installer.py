"""Install packs from a local path, GitHub, or the index; record pinned commits in a lock file.

Sources:
  ./path/to/pack                        local directory
  github:owner/repo[/subdir][@ref]      ref = tag, branch or commit (default: HEAD)
  <name>                                look up in the index; installs its pinned commit
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from autoforge import config
from autoforge.packs.errors import PackExists, PackNotFound, PackSourceError, ShaMismatch
from autoforge.packs.index import load_index
from autoforge.packs.manifest import NAME_RE, Pack, load_pack

GITHUB_RE = re.compile(r"^github:(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)(?P<subdir>(/[\w.-]+)*)(@(?P<ref>[\w./-]+))?$")


class InstalledPack(BaseModel):
    name: str
    version: str
    source: str
    commit_sha: str | None = None
    installed_at: str
    path: str


class GithubSource(BaseModel):
    owner: str
    repo: str
    subdir: str = ""
    ref: str = "HEAD"

    @property
    def url(self) -> str:
        return github_url(self.owner, self.repo)

    @property
    def unpinned(self) -> str:
        return f"github:{self.owner}/{self.repo}{'/' + self.subdir if self.subdir else ''}"


def github_url(owner: str, repo: str) -> str:
    # Indirection so tests can point at a local git repo.
    return f"https://github.com/{owner}/{repo}.git"


def parse_github(source: str) -> GithubSource:
    m = GITHUB_RE.match(source)
    if not m:
        raise PackSourceError(f"bad GitHub source: {source}", hint="Use github:owner/repo[/subdir][@ref]")
    return GithubSource(
        owner=m["owner"], repo=m["repo"], subdir=m["subdir"].strip("/"), ref=m["ref"] or "HEAD"
    )


# ---------- lock file ----------


def read_lock() -> dict[str, InstalledPack]:
    path = config.lock_file()
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: InstalledPack.model_validate(v) for k, v in raw.items()}


def _write_lock(lock: dict[str, InstalledPack]) -> None:
    path = config.lock_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: v.model_dump() for k, v in sorted(lock.items())}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------- fetching ----------


def _git(*args: str, cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=600
        ).stdout.strip()
    except FileNotFoundError:
        raise PackSourceError("git is not installed", hint="Install git to install packs from GitHub.")
    except subprocess.CalledProcessError as e:
        raise PackSourceError(f"git {args[0]} failed: {e.stderr.strip()}")


def _fetch_github(src: GithubSource, workdir: Path) -> tuple[Path, str]:
    """Shallow-fetch one ref (works for tags, branches and commit SHAs). Returns (pack dir, sha)."""
    _git("init", "-q", cwd=workdir)
    _git("fetch", "-q", "--depth", "1", src.url, src.ref, cwd=workdir)
    _git("checkout", "-q", "FETCH_HEAD", cwd=workdir)
    sha = _git("rev-parse", "HEAD", cwd=workdir)
    return workdir / src.subdir, sha


# ---------- install / remove ----------


def install(source: str, force: bool = False) -> InstalledPack:
    expected_sha = None
    if NAME_RE.fullmatch(source) and not Path(source).exists():
        index, _ = load_index()
        entry = index.get(source)
        if entry is None:
            raise PackNotFound(f"no pack named '{source}' in the index")
        source, expected_sha = entry.pinned_source, entry.commit_sha

    with tempfile.TemporaryDirectory(prefix="autoforge-pack-") as tmp:
        sha = None
        if source.startswith("github:"):
            gh = parse_github(source)
            pack_dir, sha = _fetch_github(gh, Path(tmp))
            if expected_sha and sha != expected_sha:
                raise ShaMismatch(f"expected commit {expected_sha}, got {sha}")
            recorded_source = f"{gh.unpinned}@{sha}"
        else:
            pack_dir = Path(source).expanduser().resolve()
            if not pack_dir.is_dir():
                raise PackSourceError(f"local path not found: {source}")
            recorded_source = f"local:{pack_dir}"

        pack = load_pack(pack_dir)
        return _copy_in(pack, recorded_source, sha, force)


def _copy_in(pack: Pack, source: str, sha: str | None, force: bool) -> InstalledPack:
    name = pack.manifest.name
    lock = read_lock()
    dest = config.packs_dir() / name
    if (name in lock or dest.exists()) and not force:
        raise PackExists(f"pack '{name}' is already installed ({lock[name].version if name in lock else '?'})")
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pack.root, dest, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    record = InstalledPack(
        name=name,
        version=pack.manifest.version,
        source=source,
        commit_sha=sha,
        installed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        path=str(dest),
    )
    lock[name] = record
    _write_lock(lock)
    return record


def remove(name: str) -> InstalledPack:
    lock = read_lock()
    if name not in lock:
        raise PackNotFound(f"pack '{name}' is not installed")
    record = lock.pop(name)
    shutil.rmtree(record.path, ignore_errors=True)
    _write_lock(lock)
    return record


def get_installed(name: str) -> tuple[InstalledPack, Pack]:
    lock = read_lock()
    if name not in lock:
        raise PackNotFound(f"pack '{name}' is not installed")
    return lock[name], load_pack(Path(lock[name].path))


def missing_dependencies(pack: Pack) -> list[str]:
    """Pip requirements whose distribution is not installed (version ranges not checked yet)."""
    from importlib.metadata import PackageNotFoundError, distribution

    missing = []
    for req in pack.manifest.dependencies:
        dist = re.split(r"[\s<>=!~;\[]", req, maxsplit=1)[0]
        try:
            distribution(dist)
        except PackageNotFoundError:
            missing.append(req)
    return missing
