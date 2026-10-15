import json

from typer.testing import CliRunner

from autoforge.cli.main import app

from conftest import EXAMPLE_PACK

runner = CliRunner()


def call(*args):
    r = runner.invoke(app, [*args, "--json"])
    return r.exit_code, json.loads(r.stdout)


def test_validate_ok():
    code, body = call("pack", "validate", str(EXAMPLE_PACK))
    assert code == 0 and body["data"]["name"] == "tabular-classic"


def test_validate_reports_problems(pack_copy):
    (pack_copy / "KNOWLEDGE.md").unlink()
    code, body = call("pack", "validate", str(pack_copy))
    assert code == 1 and body["error"]["code"] == "PACK_INVALID"
    assert "missing file: KNOWLEDGE.md" in body["data"]["problems"]


def test_install_info_list_remove(pack_copy):
    assert call("pack", "install", str(pack_copy))[0] == 0
    code, body = call("pack", "install", str(pack_copy))
    assert code == 1 and body["error"]["code"] == "PACK_EXISTS"
    code, body = call("pack", "info", "tabular-classic")
    assert code == 0 and "When to use" in body["data"]["knowledge"]
    assert isinstance(body["data"]["missing_dependencies"], list)
    assert [p["name"] for p in call("pack", "list")[1]["data"]] == ["tabular-classic"]
    assert call("pack", "remove", "tabular-classic")[0] == 0


def test_index_add_validate_and_search(tmp_path, monkeypatch):
    idx = tmp_path / "index.json"
    code, _ = call("index", "add", str(idx), str(EXAMPLE_PACK),
                   "--source", "github:o/r/packs/tabular-classic", "--commit", "b" * 40, "--tag", "csv")
    assert code == 0
    assert call("index", "validate", str(idx))[1]["data"]["packs"] == 1
    monkeypatch.setenv("AUTOFORGE_INDEX_URL", str(idx))
    code, body = call("pack", "search", "csv", "--task", "tabular-regression")
    assert code == 0 and body["data"]["hits"][0]["runnable_recipes"] == ["random_forest"]


def test_search_without_index_errors():
    code, body = call("pack", "search", "--offline")
    assert code == 1 and body["error"]["code"] == "INDEX_UNAVAILABLE"
