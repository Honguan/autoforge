import pytest

from autoforge.packs.errors import IndexUnavailable, PackInvalid
from autoforge.packs.index import IndexEntry, IndexRecipe, PackIndex, dump_index, load_index, parse_index, search
from autoforge.schemas.core import CPUInfo, GPUInfo, HardwareProfile

SHA = "a" * 40


def _entry(name, tasks, recipes, summary="", tags=()):
    return IndexEntry(name=name, version="1.0.0", summary=summary, source=f"github:o/{name}",
                      commit_sha=SHA, tasks=tasks, tags=list(tags), recipes=recipes)


INDEX = PackIndex(packs=[
    _entry("tabular-classic", ["tabular-classification"],
           [IndexRecipe(id="rf", task_types=["tabular-classification"], device="cpu")],
           summary="classic ML for CSV"),
    _entry("llm-lora", ["llm-sft"],
           [IndexRecipe(id="qlora", task_types=["llm-sft"], device="cuda", min_vram_gb=8)],
           summary="fine-tune LLMs with LoRA", tags=["llm", "peft"]),
])


def _hw(vram=None):
    gpus = [GPUInfo(index=0, name="RTX", vram_total_gb=vram)] if vram else []
    return HardwareProfile(os="x", python_version="3", cpu=CPUInfo(model="c", physical_cores=1, logical_cores=1),
                           ram_total_gb=8, ram_available_gb=8, disk_free_gb=10, gpus=gpus,
                           torch_cuda_available=bool(vram))


def test_search_by_task_and_query():
    assert [h.entry.name for h in search(INDEX, task="llm-sft")] == ["llm-lora"]
    assert [h.entry.name for h in search(INDEX, query="LoRA")] == ["llm-lora"]
    assert len(search(INDEX)) == 2


def test_search_fits_hardware():
    assert [h.entry.name for h in search(INDEX, hardware=_hw())] == ["tabular-classic"]
    assert [h.entry.name for h in search(INDEX, hardware=_hw(6))] == ["tabular-classic"]
    assert {h.entry.name for h in search(INDEX, hardware=_hw(12))} == {"tabular-classic", "llm-lora"}


def test_parse_index_rejects_duplicates_and_bad_sha():
    dup = PackIndex(packs=[INDEX.packs[0], INDEX.packs[0]])
    with pytest.raises(PackInvalid):
        parse_index(dump_index(dup))
    with pytest.raises(PackInvalid):
        parse_index(dump_index(INDEX).replace(SHA, "notasha", 1))


def test_load_index_falls_back_to_cache(tmp_path, monkeypatch, autoforge_home):
    monkeypatch.setenv("AUTOFORGE_INDEX_URL", "https://127.0.0.1:9/index.json")
    with pytest.raises(IndexUnavailable):
        load_index()
    cache = autoforge_home / "cache" / "index.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(dump_index(INDEX))
    index, notes = load_index()
    assert len(index.packs) == 2 and notes
