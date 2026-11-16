"""
End-to-end tests: real browser against a real server.

Covers the Q2 flows from the test plan — login lobby redirect, running a
simulation from a preset, session expiry, and the challenge loop.

Skipped automatically when Playwright or a Chromium browser is unavailable,
so the unit suite still runs anywhere.  Run just these with:

    pytest tests/e2e -v
"""

import os
import re
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
    # The unit tests run jobs by hand (conftest turns the worker off); here the
    # real worker thread runs them, as in production
    env = dict(os.environ, LUNALAB_DB_PATH=str(db), LUNALAB_JOB_WORKER="1")
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


# Pages pull Plotly (a parser-blocking <head> script), three.js and fonts from
# public CDNs.  A slow CDN made these tests flaky: Page.goto timed out after
# 30 s waiting for the page to load (seen in 1 of ~12 full runs).  The flows
# under test never depend on what those libraries draw, so the browser gets a
# no-op Plotly and the other CDN requests are dropped: no network, no flake.
EXTERNAL_CDN = re.compile(r"https://(cdn\.plot\.ly|cdn\.jsdelivr\.net|fonts\.googleapis\.com|fonts\.gstatic\.com)/")
PLOTLY_STUB = "window.Plotly = { newPlot: () => Promise.resolve() };"


def _offline_cdn(route):
    if "cdn.plot.ly" in route.request.url:
        route.fulfill(status=200, content_type="application/javascript", body=PLOTLY_STUB)
    else:
        route.abort()


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    ctx.route(EXTERNAL_CDN, _offline_cdn)
    pg = ctx.new_page()
    pg.set_default_navigation_timeout(60_000)
    pg.set_default_timeout(30_000)
    yield pg
    ctx.close()


def goto(page, url):
    page.goto(url, wait_until="domcontentloaded")


def wait_url(page, pattern):
    page.wait_for_url(pattern, wait_until="domcontentloaded")


def register(page, base, open_login=True):
    username = f"e2e-{uuid.uuid4().hex[:8]}"
    if open_login:
        goto(page, f"{base}/login")
    page.click("#tab-btn-register")
    page.fill("#auth-username", username)
    page.fill("#auth-password", "pass1234")
    page.click("#auth-submit-btn")
    return username


def test_unauthenticated_visit_redirects_to_login_and_back(page, server):
    base = server["base"]
    goto(page, f"{base}/history")
    wait_url(page, "**/login?next=%2Fhistory")
    register(page, base, open_login=False)
    wait_url(page, f"{base}/history")


def test_tycho_preset_produces_complex_crater(page, server):
    base = server["base"]
    username = register(page, base)
    wait_url(page, f"{base}/")
    goto(page, f"{base}/experiment")
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
    wait_url(page, f"{base}/")
    # Let the landing page's own /api/auth/me finish first.  Each valid request
    # slides the session expiry forward, so a request still in flight would
    # overwrite the expiry written below and revive the session (seen on CI).
    page.wait_for_load_state("networkidle")
    conn = sqlite3.connect(server["db"])
    conn.execute("UPDATE user_sessions SET expires_at = '2000-01-01T00:00:00+00:00'")
    conn.commit()
    conn.close()
    goto(page, f"{base}/experiment")
    wait_url(page, "**/login?next=%2Fexperiment")


def test_challenge_run_updates_records_table(page, server):
    base = server["base"]
    username = register(page, base)
    wait_url(page, f"{base}/")
    goto(page, f"{base}/challenge")
    page.wait_for_selector("#challenge-form")
    page.click("#btn-run-challenge")
    page.wait_for_selector("#section-evaluation:not(.hidden)", timeout=15000)
    assert page.inner_text("#eval-grade-badge") in {"S", "A", "B", "C"}
    page.wait_for_selector(f"#records-body td:has-text('{username}')")


def test_navigating_away_mid_auth_check_keeps_session(page, server):
    """Regression: leaving a page while /api/auth/me was in flight logged the user out."""
    base = server["base"]
    register(page, base)
    wait_url(page, f"{base}/")
    for path in ["/experiment", "/history", "/challenge", "/experiment"]:
        goto(page, f"{base}{path}")          # navigate immediately, mid auth check
    wait_url(page, f"{base}/experiment")
    page.wait_for_timeout(500)
    assert page.evaluate("localStorage.getItem('lunalab_token')")
    assert "/login" not in page.url


def test_login_next_param_cannot_redirect_off_site(page, server):
    """Test plan B-12: an open redirect via ?next=//evil.com must land on /."""
    base = server["base"]
    goto(page, f"{base}/login?next=//evil.example")
    register(page, base, open_login=False)
    wait_url(page, f"{base}/")


def test_username_markup_is_shown_as_text(page, server):
    """Test plan B-13: a username containing HTML is displayed, not executed."""
    base = server["base"]
    evil = f"<img src=x onerror=window.pwned=1>{uuid.uuid4().hex[:4]}"
    goto(page, f"{base}/login")
    page.click("#tab-btn-register")
    page.fill("#auth-username", evil[:30])
    page.fill("#auth-password", "pass1234")
    page.click("#auth-submit-btn")
    wait_url(page, f"{base}/")
    goto(page, f"{base}/experiment")
    page.wait_for_selector(".user-badge__name")
    assert page.inner_text(".user-badge__name") == evil[:30]
    assert page.evaluate("window.pwned") is None


# ---------------------------------------------------------------------------
# Science pages: case study, inference, surface age
# ---------------------------------------------------------------------------

def _collect_errors(page):
    """Uncaught exceptions and console errors, except the CDN requests the page fixture aborts."""
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: m.type == "error" and "net::ERR_FAILED" not in m.text and errors.append(m.text))
    return errors


def test_case_study_shows_geometry_and_findings(page, server):
    base = server["base"]
    errors = _collect_errors(page)
    register(page, base)
    wait_url(page, f"{base}/")
    goto(page, f"{base}/case-study")
    page.wait_for_selector("#cs-content:not(.hidden)")
    # The ephemeris confirms the flash was visible: two passing checks
    assert page.locator("#geometry-checks .check--pass").count() == 2
    assert "Inverting the model" in page.inner_text("#findings")
    page.select_option("#material-select", "rock")
    assert "competent rock" in page.inner_text("#chain")
    assert errors == []


def test_inference_presets_fill_the_form_and_run(page, server):
    base = server["base"]
    errors = _collect_errors(page)
    register(page, base)
    wait_url(page, f"{base}/")
    goto(page, f"{base}/inference")
    page.wait_for_selector("#inf-content:not(.hidden)")
    # The default preset is the crater alone: no luminous-efficiency card
    assert "Luminous efficiency" not in page.inner_text("#headline")
    page.click(".preset[data-id='2013-flash']")
    page.wait_for_function("document.querySelector('#headline').textContent.includes('Luminous efficiency')")
    assert page.input_value("#in-light") == "7100000"
    page.click(".preset[data-id='copernicus']")
    page.wait_for_function("document.querySelector('#headline-note').textContent.includes('competent-rock')")
    assert errors == []


def test_inference_custom_crater(page, server):
    base = server["base"]
    register(page, base)
    wait_url(page, f"{base}/")
    goto(page, f"{base}/inference")
    page.wait_for_selector("#inf-content:not(.hidden)")
    page.fill("#in-diameter", "500")
    page.fill("#in-light", "")
    page.fill("#in-velocity", "")
    page.click("#btn-infer")
    page.wait_for_function("document.querySelector('#headline-note').textContent.startsWith('500 m')")
    assert page.locator(".preset--active").count() == 0


def test_surface_age_simulation_and_time_slider(page, server):
    base = server["base"]
    errors = _collect_errors(page)
    register(page, base)
    wait_url(page, f"{base}/")
    goto(page, f"{base}/surface-age")
    page.wait_for_selector("#fit-rows tr")
    today = page.inner_text("#clock")
    assert today.startswith("today") and int(re.search(r"([\d,]+) craters", today).group(1).replace(",", "")) > 1000
    # Back to the moment the surface formed: no craters yet
    page.fill("#time-slider", "0")
    page.dispatch_event("#time-slider", "input")
    assert re.search(r"· 0 craters", page.inner_text("#clock"))
    # A younger surface, re-simulated
    page.fill("#age-slider", "2")
    page.dispatch_event("#age-slider", "change")
    page.wait_for_function("document.querySelector('#fit-rows').textContent.includes('True age: 2.0')")
    assert errors == []


def test_surface_age_count_area_toggle_and_site_map(page, server):
    base = server["base"]
    errors = _collect_errors(page)
    register(page, base)
    wait_url(page, f"{base}/")
    goto(page, f"{base}/surface-age")
    page.wait_for_function("document.querySelector('#sites-finding').textContent.length > 0")
    assert page.inner_text("#sites-finding").startswith("Every crater within 150 km")
    page.click(".toggle__btn[data-area='unit']")
    page.wait_for_function("document.querySelector('#sites-finding').textContent.startsWith('On the mapped unit')")
    page.select_option("#site-select", "apollo17")
    page.wait_for_timeout(500)
    assert errors == []


def test_every_page_links_to_the_science_pages(page, server):
    base = server["base"]
    register(page, base)
    wait_url(page, f"{base}/")
    for path in ["/", "/experiment", "/challenge", "/history", "/case-study", "/inference", "/surface-age"]:
        goto(page, f"{base}{path}")
        links = page.eval_on_selector_all(".navbar__links a", "els => els.map(e => e.getAttribute('href'))")
        assert {"/case-study", "/inference", "/surface-age"} <= set(links), path


def test_inference_shows_the_last_preset_clicked(page, server):
    """Regression: a slow earlier response (the flash preset) overwrote a later one."""
    base = server["base"]
    register(page, base)
    wait_url(page, f"{base}/")
    goto(page, f"{base}/inference")
    page.wait_for_selector("#inf-content:not(.hidden)")
    page.click(".preset[data-id='2013-flash']")      # ~1M samples: the slow one
    page.click(".preset[data-id='linne']")           # 300k samples: answers first
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(300)
    note = page.inner_text("#headline-note")
    assert note.startswith("2,200 m") and "flash" not in note


def test_batch_job_runs_in_the_background_and_shows_its_result(page, server):
    base = server["base"]
    register(page, base)
    wait_url(page, f"{base}/")
    goto(page, f"{base}/batch")
    page.wait_for_selector("#jobs-empty:not(.hidden)")

    page.fill("#in-diameters", "18.8\n43, 2000")
    assert page.inner_text("#diameter-hint").startswith("3 craters")
    page.click("#btn-submit")

    # The worker thread picks the job up; the page polls until it is done
    page.wait_for_selector("#jobs-body tr [data-status]:text-is('succeeded')")
    page.wait_for_selector("#result-content:not(.hidden)")
    assert page.locator("#result-body tr").count() == 3
    assert page.inner_text("#result-body tr:nth-child(3) td:nth-child(2)") == "Rock"
    assert page.locator("#jobs-body button[data-action='cancel']").count() == 0

    # The job belongs to the account, not the page: it is still there after a reload
    page.reload(wait_until="domcontentloaded")
    page.wait_for_selector("#result-content:not(.hidden)")
    assert page.locator("#jobs-body tr").count() == 1


def test_outdated_batch_result_is_flagged_and_can_be_run_again(page, server):
    import sqlite3
    base = server["base"]
    register(page, base)
    wait_url(page, f"{base}/")
    goto(page, f"{base}/batch")
    page.wait_for_selector("#jobs-empty:not(.hidden)")
    page.fill("#in-diameters", "18.8")
    page.fill("#in-seed", "3")
    page.click("#btn-submit")
    page.wait_for_selector("#result-content:not(.hidden)")
    assert "seed 3" in page.inner_text("#result-provenance")
    assert page.locator("[data-outdated]").count() == 0
    first_row = page.inner_text("#result-body tr")

    # As if the model had been changed since the job ran
    conn = sqlite3.connect(server["db"])
    conn.execute("UPDATE jobs SET provenance_json = ?", ('{"fingerprint": "older", "numpy": "1.0"}',))
    conn.commit()
    conn.close()
    page.reload(wait_until="domcontentloaded")
    page.wait_for_selector("[data-outdated]")
    page.wait_for_selector("#result-outdated:not(.hidden)")

    page.click("#jobs-body button[data-action='rerun']")
    page.wait_for_function("document.querySelectorAll('#jobs-body tr').length === 2")
    page.wait_for_selector("#result-content:not(.hidden) >> #result-provenance:has-text('seed 3')")
    page.wait_for_selector("#jobs-body tr:first-child [data-status]:text-is('succeeded')")
    # The new job is current; the old one is still listed, still flagged
    assert page.locator("#jobs-body tr:first-child [data-outdated]").count() == 0
    assert page.locator("#jobs-body tr:nth-child(2) [data-outdated]").count() == 1
    page.wait_for_selector("#result-outdated", state="hidden")
    assert page.inner_text("#result-body tr") == first_row   # same seed, same numbers


def test_batch_rejects_a_bad_list_before_submitting(page, server):
    base = server["base"]
    register(page, base)
    wait_url(page, f"{base}/")
    goto(page, f"{base}/batch")
    page.wait_for_selector("#jobs-empty:not(.hidden)")

    page.fill("#in-diameters", "18.8\nbig")
    assert "not a positive number" in page.inner_text("#diameter-hint")
    page.click("#btn-submit")
    page.wait_for_selector(".toast")
    assert page.locator("#jobs-body tr").count() == 0


def test_batch_loads_craters_around_a_landing_site(page, server):
    base = server["base"]
    register(page, base)
    wait_url(page, f"{base}/")
    goto(page, f"{base}/batch")
    page.wait_for_selector("#in-site option", state="attached")   # options are never "visible"

    page.select_option("#in-site", "change5")
    page.select_option("#in-radius", "25")
    page.click("#btn-load-site")
    page.wait_for_function("document.getElementById('in-diameters').value.length > 0")
    # Catalogue diameters are in km; the job takes metres
    first = float(page.input_value("#in-diameters").split()[0])
    assert first >= 1000
    assert "crater" in page.inner_text("#diameter-hint")
