"""
End-to-end tests: real browser against a real server.

Covers the Q2 flows from the test plan — login lobby redirect, running a
simulation from a preset, session expiry, and the challenge loop.

Skipped automatically when Playwright or a Chromium browser is unavailable,
so the unit suite still runs anywhere.  Run just these with:

    pytest tests/e2e -v
"""

import os
import socket
import subprocess
import sys
import time
import uuid

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    port = _free_port()
    db = tmp_path_factory.mktemp("e2e") / "e2e.db"
    env = dict(os.environ, LUNALAB_DB_PATH=str(db))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api:app", "--port", str(port)],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    yield {"base": base, "db": str(db)}
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="module")
def browser():
    with playwright_sync.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception:
            if not os.path.exists(MAC_CHROME):
                pytest.skip("No Chromium available (run `playwright install chromium`)")
            b = p.chromium.launch(executable_path=MAC_CHROME)
        yield b
        b.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    pg = ctx.new_page()
    yield pg
    ctx.close()


def register(page, base, open_login=True):
    username = f"e2e-{uuid.uuid4().hex[:8]}"
    if open_login:
        page.goto(f"{base}/login")
    page.click("#tab-btn-register")
    page.fill("#auth-username", username)
    page.fill("#auth-password", "pass1234")
    page.click("#auth-submit-btn")
    return username


def test_unauthenticated_visit_redirects_to_login_and_back(page, server):
    base = server["base"]
    page.goto(f"{base}/history")
    page.wait_for_url("**/login?next=%2Fhistory")
    register(page, base, open_login=False)
    page.wait_for_url(f"{base}/history")


def test_tycho_preset_produces_complex_crater(page, server):
    base = server["base"]
    username = register(page, base)
    page.wait_for_url(f"{base}/")
    page.goto(f"{base}/experiment")
    page.click(".preset-card:has-text('Tycho')")
    page.click("#btn-run")
    page.wait_for_selector("#section-results:not(.hidden)", timeout=15000)
    assert page.inner_text("#result-crater-type") == "Complex crater"
    assert "km" in page.inner_text("#res-crater-diam")
    # +15 XP shows up in the navbar badge
    page.wait_for_function("document.querySelector('.user-badge__xp')?.textContent.startsWith('15')")
    assert username in page.inner_text(".user-badge")


def test_expired_session_returns_to_login(page, server):
    import sqlite3
    base = server["base"]
    register(page, base)
    page.wait_for_url(f"{base}/")
    conn = sqlite3.connect(server["db"])
    conn.execute("UPDATE user_sessions SET expires_at = '2000-01-01T00:00:00+00:00'")
    conn.commit()
    conn.close()
    page.goto(f"{base}/experiment")
    page.wait_for_url("**/login?next=%2Fexperiment")


def test_challenge_run_updates_records_table(page, server):
    base = server["base"]
    username = register(page, base)
    page.wait_for_url(f"{base}/")
    page.goto(f"{base}/challenge")
    page.wait_for_selector("#challenge-form")
    page.click("#btn-run-challenge")
    page.wait_for_selector("#section-evaluation:not(.hidden)", timeout=15000)
    assert page.inner_text("#eval-grade-badge") in {"S", "A", "B", "C"}
    page.wait_for_selector(f"#records-body td:has-text('{username}')")
