"""`autoforge pack ...` and `autoforge index ...` commands."""

from __future__ import annotations

from pathlib import Path

import typer

from autoforge.cli.output import JsonOpt, emit, errors_as_envelope, fail
from autoforge.packs import installer
from autoforge.packs.index import PackIndex, dump_index, entry_from_pack, load_index, parse_index, search
from autoforge.packs.manifest import collect_problems, load_pack

pack_app = typer.Typer(help="Install, inspect and search method packs.", no_args_is_help=True)
index_app = typer.Typer(help="Maintain a pack index (index.json).", no_args_is_help=True)


@pack_app.command("validate")
def validate(path: Path, as_json: bool = JsonOpt) -> None:
    """Check a pack directory without executing any of its code."""
    pack, problems = collect_problems(path)
    if pack is None:
        fail(as_json, "PACK_INVALID", f"{len(problems)} problem(s)", None, {"problems": problems})
    emit(as_json, {"name": pack.manifest.name, "version": pack.manifest.version,
                   "recipes": [r.id for r in pack.recipes]},
         f"OK: {pack.manifest.name} {pack.manifest.version} ({len(pack.recipes)} recipes)")


@pack_app.command("install")
def install(
    source: str = typer.Argument(help="index name, ./local/path, or github:owner/repo[/subdir][@ref]"),
    force: bool = typer.Option(False, "--force", help="Reinstall if already installed."),
    as_json: bool = JsonOpt,
) -> None:
    """Install a pack and pin its commit in the lock file."""
    with errors_as_envelope(as_json):
        rec = installer.install(source, force=force)
    emit(as_json, rec.model_dump(), f"Installed {rec.name} {rec.version} from {rec.source}")


@pack_app.command("remove")
def remove(name: str, as_json: bool = JsonOpt) -> None:
    """Uninstall a pack."""
    with errors_as_envelope(as_json):
        rec = installer.remove(name)
    emit(as_json, rec.model_dump(), f"Removed {rec.name}")


@pack_app.command("list")
def list_installed(as_json: bool = JsonOpt) -> None:
    """List installed packs."""
    lock = installer.read_lock()
    rows = [r.model_dump() for r in lock.values()]
    human = "\n".join(f"{r.name:<28} {r.version:<10} {r.source}" for r in lock.values()) or "(no packs installed)"
    emit(as_json, rows, human)


@pack_app.command("info")
def info(
    name: str,
    knowledge: bool = typer.Option(True, "--knowledge/--no-knowledge", help="Include KNOWLEDGE.md text."),
    as_json: bool = JsonOpt,
) -> None:
    """Show an installed pack's manifest, recipes, missing deps and KNOWLEDGE.md (for AI reading)."""
    with errors_as_envelope(as_json):
        rec, pack = installer.get_installed(name)
    data = {
        "installed": rec.model_dump(),
        "manifest": pack.manifest.model_dump(),
        "recipes": [r.model_dump() for r in pack.recipes],
        "missing_dependencies": installer.missing_dependencies(pack),
    }
    if knowledge:
        data["knowledge"] = pack.knowledge()
    emit(as_json, data)


@pack_app.command("search")
def search_cmd(
    query: str = typer.Argument(None, help="Free-text keywords."),
    task: str = typer.Option(None, "--task", help="Exact task type, e.g. tabular-classification."),
    fit: bool = typer.Option(False, "--fit", help="Only recipes this machine's hardware can run."),
    offline: bool = typer.Option(False, "--offline", help="Use the cached index only."),
    as_json: bool = JsonOpt,
) -> None:
    """Search the pack index."""
    hw = None
    if fit:
        from autoforge.hardware.detect import detect_hardware

        hw = detect_hardware()
    with errors_as_envelope(as_json):
        index, notes = load_index(refresh=not offline)
    hits = search(index, task=task, query=query, hardware=hw)
    lock = installer.read_lock()
    rows = [
        h.model_dump(mode="json") | {"installed": h.entry.name in lock, "install_with": h.entry.name}
        for h in hits
    ]
    human = "\n".join(
        f"{h.entry.name:<28} {h.entry.version:<8} {', '.join(h.runnable_recipes):<30} {h.entry.summary}"
        for h in hits
    ) or "(no matches)"
    if as_json:
        emit(True, {"hits": rows, "notes": notes})
    else:
        emit(False, human="\n".join([*(f"note: {n}" for n in notes), human]))


@index_app.command("entry")
def index_entry(
    path: Path,
    source: str = typer.Option(..., help="github:owner/repo[/subdir] where the pack lives"),
    commit: str = typer.Option(..., help="40-char commit SHA to pin"),
    tag: list[str] = typer.Option([], "--tag", help="Search tag; repeatable."),
    as_json: bool = JsonOpt,
) -> None:
    """Print the index.json entry for a local pack checkout."""
    with errors_as_envelope(as_json):
        pack = load_pack(path)
    entry = entry_from_pack(pack, source, commit, tag)
    emit(as_json, entry.model_dump(mode="json"))


@index_app.command("add")
def index_add(
    index_file: Path,
    path: Path,
    source: str = typer.Option(...),
    commit: str = typer.Option(...),
    tag: list[str] = typer.Option([], "--tag"),
    as_json: bool = JsonOpt,
) -> None:
    """Add or replace a pack's entry in an index.json file."""
    with errors_as_envelope(as_json):
        pack = load_pack(path)
        index = parse_index(index_file.read_text(encoding="utf-8")) if index_file.exists() else PackIndex()
    entry = entry_from_pack(pack, source, commit, tag)
    index.packs = sorted([p for p in index.packs if p.name != entry.name] + [entry], key=lambda p: p.name)
    index_file.write_text(dump_index(index), encoding="utf-8")
    emit(as_json, entry.model_dump(mode="json"), f"Wrote {entry.name} {entry.version} to {index_file}")


@index_app.command("validate")
def index_validate(index_file: Path, as_json: bool = JsonOpt) -> None:
    """Validate an index.json file (structure, SHAs, duplicate names)."""
    with errors_as_envelope(as_json):
        index = parse_index(index_file.read_text(encoding="utf-8"))
    emit(as_json, {"packs": len(index.packs)}, f"OK: {len(index.packs)} pack(s)")


@pack_app.command("new")
def new(
    name: str,
    task: str = typer.Option(..., "--task", help="Task type, e.g. text-classification."),
    dest: Path = typer.Option(Path("."), "--dest", help="Parent directory."),
    as_json: bool = JsonOpt,
) -> None:
    """Create a valid pack skeleton (pack.yaml, KNOWLEDGE.md, recipe, trainer, smoke test)."""
    from autoforge.packs.authoring import scaffold

    with errors_as_envelope(as_json):
        res = scaffold(name, task, dest)
    emit(as_json, res, f"Created {res['path']}\nNext: edit trainers/train.py and KNOWLEDGE.md, then "
                       f"`autoforge pack test {res['path']}`")


@pack_app.command("test")
def test(
    target: str = typer.Argument(help="Installed pack name or a pack directory."),
    timeout: float = typer.Option(600, "--timeout"),
    as_json: bool = JsonOpt,
) -> None:
    """Run a pack's smoke tests (tests/) with pytest."""
    from autoforge.packs.authoring import test_pack

    path = target
    if not Path(target).is_dir():
        with errors_as_envelope(as_json):
            path = installer.get_installed(target)[0].path
    with errors_as_envelope(as_json):
        res = test_pack(path, timeout)
    if not res["ok"]:
        fail(as_json, "PACK_TEST_FAILED", res["summary"] or "tests failed", "read output_tail", res)
    emit(as_json, res, f"OK: {res['summary']} ({res['seconds']} s)")
