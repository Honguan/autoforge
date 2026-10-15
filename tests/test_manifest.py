from autoforge.packs.manifest import collect_problems, load_pack

from conftest import EXAMPLE_PACK


def test_example_pack_is_valid():
    pack = load_pack(EXAMPLE_PACK)
    assert pack.manifest.name == "tabular-classic"
    assert {r.id for r in pack.recipes} == {"logistic_regression", "random_forest"}
    assert "When to use" in pack.knowledge()


def test_missing_knowledge(pack_copy):
    (pack_copy / "KNOWLEDGE.md").unlink()
    pack, problems = collect_problems(pack_copy)
    assert pack is None and "missing file: KNOWLEDGE.md" in problems


def test_bad_name_and_version(pack_copy):
    y = pack_copy / "pack.yaml"
    import re
    text = y.read_text().replace("name: tabular-classic", "name: Tabular_Classic")
    y.write_text(re.sub(r"(?m)^version: .*$", "version: v1", text))
    _, problems = collect_problems(pack_copy)
    assert any("name" in p for p in problems) and any("version" in p for p in problems)


def test_entrypoint_function_missing(pack_copy):
    r = pack_copy / "recipes" / "random_forest.yaml"
    r.write_text(r.read_text().replace(":train_random_forest", ":nope"))
    _, problems = collect_problems(pack_copy)
    assert any("function 'nope' not defined" in p for p in problems)


def test_entrypoint_cannot_escape_pack(pack_copy):
    r = pack_copy / "recipes" / "random_forest.yaml"
    r.write_text(r.read_text().replace("trainers/sklearn_trainers.py", "../../evil.py"))
    _, problems = collect_problems(pack_copy)
    assert any("escapes" in p for p in problems)


def test_undeclared_task_type(pack_copy):
    r = pack_copy / "recipes" / "random_forest.yaml"
    r.write_text(r.read_text().replace("tabular-regression]", "image-classification]"))
    _, problems = collect_problems(pack_copy)
    assert any("not declared" in p for p in problems)
