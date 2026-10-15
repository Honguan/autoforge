import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLE_PACK = REPO / "packs" / "tabular-classic"


@pytest.fixture(autouse=True)
def autoforge_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("AUTOFORGE_HOME", str(home))
    monkeypatch.setenv("AUTOFORGE_INDEX_URL", str(tmp_path / "no-index.json"))
    return home


@pytest.fixture
def pack_copy(tmp_path):
    dest = tmp_path / "src" / "tabular-classic"
    shutil.copytree(EXAMPLE_PACK, dest)
    return dest


@pytest.fixture
def git_pack_repo(tmp_path, monkeypatch):
    """A local git repo holding the example pack under packs/, served as github:me/packs."""
    repo = tmp_path / "remote"
    shutil.copytree(EXAMPLE_PACK, repo / "packs" / "tabular-classic")
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    run("init", "-q", "-b", "main")
    run("-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    sha = run("rev-parse", "HEAD")
    monkeypatch.setattr("autoforge.packs.installer.github_url", lambda owner, name: str(repo))
    return repo, sha
