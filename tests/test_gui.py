import glob
import os
import re
import threading
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from autoforge.gui.app import create_app
from autoforge.packs import installer
from autoforge.packs.index import PackIndex, dump_index, entry_from_pack
from autoforge.packs.manifest import load_pack

REPO = Path(__file__).resolve().parents[1]
TOKEN = "test-token"


def client(token=TOKEN):
    c = TestClient(create_app(TOKEN), base_url="http://127.0.0.1")
    c.headers["X-AutoForge-Token"] = token
    return c


def test_token_and_host_are_required():
    assert client().get("/api/info").json()["ok"]
    r = client("wrong").get("/api/info")
    assert r.status_code == 401 and r.json()["error"]["code"] == "UNAUTHORIZED"
    evil = TestClient(create_app(TOKEN), base_url="http://evil.example")
    r = evil.get("/api/info", headers={"X-AutoForge-Token": TOKEN})
    assert r.status_code == 403 and r.json()["error"]["code"] == "FORBIDDEN_HOST"
    page = client().get("/")
    assert TOKEN in page.text and "frame-ancestors 'none'" in page.headers["content-security-policy"]


def test_upload_profile_and_errors():
    c = client()
    up = c.post("/api/data/upload", files={"file": ("../../evil name.csv", b"a,b\n1,x\n2,y\n")}).json()
    assert up["ok"] and up["data"]["name"] == "evil_name.csv" and "/workspace/data/" in up["data"]["path"]
    prof = c.get("/api/data/profile", params={"path": up["data"]["path"]}).json()
    assert prof["data"]["rows"] == 2 and prof["data"]["access"] == "full"
    assert [f["name"] for f in c.get("/api/data/files").json()["data"]] == ["evil_name.csv"]
    r = c.get("/api/runs/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "RUN_NOT_FOUND"
    r = c.post("/api/runs", json={"plan": {"task": 1}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "PLAN_INVALID"


def test_series_only_contains_real_curves(tmp_path):
    from autoforge import api

    installer.install(str(Path(__file__).parent / "fixtures" / "engine-pack"))
    from autoforge.engine import runner
    from autoforge.schemas.core import TaskSpec, TrainingPlan
    data = tmp_path / "d.txt"
    data.write_text("x")
    plan = TrainingPlan(task=TaskSpec(task_type="toy", input_modality="x", output_modality="y", metric="loss"),
                        data_path=str(data), pack="engine-test", recipe="slow", hyperparameters={"steps": 4})
    s = runner.create_run(plan)
    runner.start(s)
    runner.wait(s, 30)
    assert set(api.run_series(s.run_id)) == {"loss"} and len(api.run_series(s.run_id)["loss"]) == 4


def _chromium():
    pw = pytest.importorskip("playwright.sync_api")
    exe = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))
    return pw, (exe[-1] if exe else None)


def test_gui_end_to_end_without_cli(tmp_path, monkeypatch):
    """P8 acceptance: upload → task → method → validate → train → register → predict → tune, all in the browser."""
    pytest.importorskip("sklearn")
    pw, exe = _chromium()
    import uvicorn

    pack = REPO / "packs" / "tabular-classic"
    idx = tmp_path / "index.json"
    idx.write_text(dump_index(PackIndex(packs=[entry_from_pack(load_pack(pack), "github:x/y/tabular-classic", "0" * 40)])))
    monkeypatch.setenv("AUTOFORGE_INDEX_URL", str(idx))
    installer.install(str(pack))

    port = 8830 + os.getpid() % 100
    server = uvicorn.Server(uvicorn.Config(create_app("e2e"), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    try:
        with pw.sync_playwright() as p:
            try:
                browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
            except Exception as e:  # no browser in this environment
                pytest.skip(f"chromium not available: {e}")
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            expect = pw.expect
            page.goto(f"http://127.0.0.1:{port}/#/new")
            page.set_input_files("#upload", str(REPO / "examples" / "data" / "churn.csv"))
            page.click("#upload-btn")
            expect(page.get_by_text("資料概況")).to_be_visible()
            page.click("#next-1")
            page.select_option("#task", "tabular-classification")
            page.select_option("#target", "churned")
            page.click("#next-2")
            page.locator('[data-recipe="logistic_regression"]').click()
            expect(page.locator('[data-param="C"]')).to_be_visible()
            page.click("#next-3")
            expect(page.get_by_text("檢查通過")).to_be_visible(timeout=20_000)
            page.click("#start")
            page.wait_for_url(re.compile(r"#/runs/"))
            expect(page.locator(".badge.succeeded")).to_be_visible(timeout=60_000)
            expect(page.get_by_text("nullnull")).to_have_count(0)
            run_url = page.url
            page.fill("#model-name", "churn")
            page.click("#register")
            page.wait_for_url(re.compile(r"#/models/churn"))
            for k, v in {"tenure_months": "2", "monthly_fee": "110", "contract": "month-to-month", "support_calls": "5"}.items():
                page.fill(f'[data-feature="{k}"]', v)
            page.click("#predict")
            expect(page.locator("#prediction")).to_have_text(re.compile(r"^[01]$"))
            page.goto(run_url)
            page.fill("#hpo-trials", "3")
            page.click("#hpo-start")
            page.wait_for_url(re.compile(r"#/hpo/"))
            expect(page.locator(".badge.succeeded")).to_be_visible(timeout=90_000)
            expect(page.locator("#best-value")).not_to_have_text("–")
            assert errors == [], errors
            browser.close()
    finally:
        server.should_exit = True
