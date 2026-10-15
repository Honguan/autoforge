import json

import pytest

from autoforge.packs import installer
from autoforge.packs.errors import PackExists, PackInvalid, PackNotFound, PackSourceError, ShaMismatch
from autoforge.packs.index import PackIndex, dump_index, entry_from_pack
from autoforge.packs.manifest import load_pack


def test_install_local_then_list_and_remove(pack_copy):
    rec = installer.install(str(pack_copy))
    assert rec.name == "tabular-classic" and rec.source.startswith("local:")
    assert "tabular-classic" in installer.read_lock()
    with pytest.raises(PackExists):
        installer.install(str(pack_copy))
    installer.install(str(pack_copy), force=True)
    installer.remove("tabular-classic")
    assert installer.read_lock() == {}
    with pytest.raises(PackNotFound):
        installer.remove("tabular-classic")


def test_install_invalid_pack_is_rejected(pack_copy):
    (pack_copy / "pack.yaml").write_text("name: x\n")
    with pytest.raises(PackInvalid):
        installer.install(str(pack_copy))
    assert installer.read_lock() == {}


def test_parse_github():
    s = installer.parse_github("github:Honguan/autoforge/packs/tabular-classic@v1.0.0")
    assert (s.owner, s.repo, s.subdir, s.ref) == ("Honguan", "autoforge", "packs/tabular-classic", "v1.0.0")
    assert installer.parse_github("github:a/b").ref == "HEAD"
    with pytest.raises(PackSourceError):
        installer.parse_github("github:only-owner")


def test_install_from_git_pins_sha(git_pack_repo):
    _, sha = git_pack_repo
    rec = installer.install("github:me/remote/packs/tabular-classic@main")
    assert rec.commit_sha == sha
    assert rec.source == f"github:me/remote/packs/tabular-classic@{sha}"


def _write_index(tmp_path, monkeypatch, sha):
    from conftest import EXAMPLE_PACK

    entry = entry_from_pack(load_pack(EXAMPLE_PACK), "github:me/remote/packs/tabular-classic", sha)
    path = tmp_path / "index.json"
    path.write_text(dump_index(PackIndex(packs=[entry])))
    monkeypatch.setenv("AUTOFORGE_INDEX_URL", str(path))


def test_install_by_index_name(git_pack_repo, tmp_path, monkeypatch):
    _, sha = git_pack_repo
    _write_index(tmp_path, monkeypatch, sha)
    rec = installer.install("tabular-classic")
    assert rec.commit_sha == sha


def test_install_by_index_name_detects_sha_mismatch(git_pack_repo, tmp_path, monkeypatch):
    repo, sha = git_pack_repo
    _write_index(tmp_path, monkeypatch, sha)
    # Pretend the index pins this SHA but the fetched commit differs.
    monkeypatch.setattr(installer, "_fetch_github", lambda src, wd: (repo / "packs" / "tabular-classic", "f" * 40))
    with pytest.raises(ShaMismatch):
        installer.install("tabular-classic")


def test_unknown_index_name(tmp_path, monkeypatch):
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"schema_version": 1, "packs": []}))
    monkeypatch.setenv("AUTOFORGE_INDEX_URL", str(path))
    with pytest.raises(PackNotFound):
        installer.install("does-not-exist")
