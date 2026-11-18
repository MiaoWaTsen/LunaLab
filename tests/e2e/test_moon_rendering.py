"""
The 3D Moon as rendered: texture registration on screen and the terminator.

Unlike the user-flow tests these need three.js and WebGL, so three.js comes
from its CDN (skipped if unreachable) and Chromium renders with SwiftShader.
Each test swaps in a synthetic texture so it measures one thing: coloured
landmark dots for registration, a flat grey Moon for the lighting model.

A Moon page that never finishes loading looks the same as an unreachable
CDN, so these tests skip only locally: in CI (CI=true) they fail instead,
as the PostgreSQL tests do (tests/conftest.py).
"""

import io
import math
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tests.e2e.test_user_flows import MAC_CHROME, goto, playwright_sync, register, server, wait_url  # noqa: F401

# A fixed moment, so the view (libration) and the Sun are reproducible:
# 2026-11-17 12:00 UTC is first quarter, the terminator crosses the near side.
AT = "2026-11-17T12:00:00Z"

LANDMARKS = {   # lat, lon (°), dot colour
    "Copernicus": (9.62, -20.08, (255, 0, 0)),
    "Tycho": (-43.31, -11.36, (0, 255, 0)),
    "Kepler": (8.12, -38.01, (0, 0, 255)),
    "Aristarchus": (23.73, -47.49, (255, 255, 0)),
    "Plato": (51.62, -9.38, (255, 0, 255)),
    "Theophilus": (-11.45, 26.28, (0, 255, 255)),
}
DOT_RADIUS_DEG = 0.6


def _png(image):
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def landmark_texture(width=4096):
    """Grey equirectangular map with a round coloured dot at each landmark."""
    height = width // 2
    img = Image.new("RGB", (width, height), (150, 150, 150))
    d = ImageDraw.Draw(img)
    for lat, lon, colour in LANDMARKS.values():
        x = (lon + 180) / 360 * width
        y = (90 - lat) / 180 * height
        ry = DOT_RADIUS_DEG / 180 * height
        rx = ry / math.cos(math.radians(lat))        # round on the sphere, not on the map
        d.ellipse((x - rx, y - ry, x + rx, y + ry), fill=colour)
    return _png(img)


def skip_unless_ci(reason):
    if os.environ.get("CI"):
        pytest.fail(f"{reason}: the 3D Moon tests must run in CI")
    pytest.skip(reason)


FLAT_NORMAL = _png(Image.new("RGB", (4, 2), (128, 128, 255)))
GREY = _png(Image.new("RGB", (512, 256), (150, 150, 150)))


@pytest.fixture(scope="module")
def webgl_browser():
    with playwright_sync.sync_playwright() as p:
        args = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
        try:
            b = p.chromium.launch(args=args)
        except Exception:
            if not os.path.exists(MAC_CHROME):
                skip_unless_ci("No Chromium available (run `playwright install chromium`)")
            b = p.chromium.launch(executable_path=MAC_CHROME, args=args)
        yield b
        b.close()


def open_moon(browser, base, color_png, dpr=1, even_light=False):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900}, device_scale_factor=dpr)
    page = ctx.new_page()
    page.set_default_timeout(60_000)
    page.route("**/static/textures/moon_color_*.jpg", lambda r: r.fulfill(status=200, content_type="image/png", body=color_png))
    page.route("**/static/textures/moon_normal_4k.jpg", lambda r: r.fulfill(status=200, content_type="image/png", body=FLAT_NORMAL))
    page.route("**/api/moon/illumination", lambda r: r.continue_(url=f"{base}/api/moon/illumination?at={AT}"))
    register(page, base)
    wait_url(page, f"{base}/")    # registering redirects home; navigating sooner is interrupted
    try:
        goto(page, f"{base}/moon")
        page.wait_for_selector("#loading-overlay", state="hidden")
    except playwright_sync.TimeoutError:
        ctx.close()
        skip_unless_ci("the Moon page did not finish loading (error in moon.js, CDN unreachable or no WebGL)")
    page.wait_for_function("document.getElementById('sun-info').textContent.includes('lit')")
    if even_light:
        page.click("#btn-sun-toggle")
    page.wait_for_timeout(1500)          # textures uploaded, a few frames drawn
    return ctx, page


def canvas_pixels(page):
    """
    The canvas as an image, and the page position (CSS px) of the image's corner.

    An element screenshot is widened to whole CSS pixels, so when the canvas
    sits at y = 351.66 the image starts at 351: without this origin every
    position read off the image is off by up to a pixel (~0.3° on the Moon).
    """
    box = page.locator("#moon-container canvas").bounding_box()
    origin = (math.floor(box["x"]), math.floor(box["y"]))
    png = page.locator("#moon-container canvas").screenshot()
    return origin, np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), float)


def pick(page, x, y):
    """Click a page point (CSS px) and read the coordinates the page reports."""
    page.mouse.click(x, y)
    lat = float(page.text_content("#val-lat").rstrip("°"))
    lon = float(page.text_content("#val-lon").rstrip("°"))
    return lat, lon


def test_landmarks_appear_where_the_page_says_they_are(webgl_browser, server):
    """
    Each landmark's dot, found on screen, is clicked: the coordinates the page
    reports must be the landmark's within 0.5°.  This covers the texture
    mapping, the sphere's orientation, the camera and the picking maths.
    """
    ctx, page = open_moon(webgl_browser, server["base"], landmark_texture(), dpr=2, even_light=True)
    (ox, oy), img = canvas_pixels(page)
    scale = 2                                            # device px per CSS px
    errors = {}
    for name, (lat, lon, colour) in LANDMARKS.items():
        c = np.array(colour, float) / 255
        # the dot's pixels: saturated, with the dot's hue (shading changes only brightness)
        norm = img / np.maximum(img.max(axis=2, keepdims=True), 1)
        mask = (np.abs(norm - c).max(axis=2) < 0.25) & (img.max(axis=2) - img.min(axis=2) > 60)
        ys, xs = np.nonzero(mask)
        assert len(xs) > 20, f"{name} dot not visible"
        got_lat, got_lon = pick(page, ox + (xs.mean() + 0.5) / scale, oy + (ys.mean() + 0.5) / scale)
        errors[name] = max(abs(got_lat - lat), abs(got_lon - lon) * math.cos(math.radians(lat)))
    ctx.close()
    assert max(errors.values()) < 0.5, errors


def test_terminator_matches_the_ephemeris(webgl_browser, server):
    """
    On a flat grey Moon (no texture, no relief) in real sunlight, brightness
    along the screen's horizontal midline must follow cos(solar incidence) and
    fall to the night-side level at 90° ± 1°, with incidence computed from the
    ephemeris's sub-solar point at each picked lat/lon.
    """
    ctx, page = open_moon(webgl_browser, server["base"], GREY)
    sun = page.evaluate(f"() => apiGet('/api/moon/illumination?at={AT}')")["sub_solar"]
    (ox, oy), img = canvas_pixels(page)
    height, width = img.shape[:2]
    y = height // 2
    lit = np.nonzero(img[y].max(axis=1) > 0)[0]
    s_lat, s_lon = math.radians(sun["latitude"]), math.radians(sun["longitude"])
    samples = []
    for x in range(lit.min() + 6, lit.max() - 6, 6):
        lat, lon = map(math.radians, pick(page, ox + x + 0.5, oy + y + 0.5))
        cos_i = math.sin(lat) * math.sin(s_lat) + math.cos(lat) * math.cos(s_lat) * math.cos(lon - s_lon)
        srgb = img[y, x].mean() / 255
        linear = srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4
        samples.append((cos_i, linear))
    ctx.close()
    cos_i, linear = np.array(samples).T
    night = np.median(linear[cos_i < -0.05])
    day = (cos_i > 0.15) & (cos_i < 0.9)
    slope, intercept = np.polyfit(cos_i[day], linear[day], 1)
    terminator = math.degrees(math.acos((night - intercept) / slope))
    assert abs(terminator - 90) < 1.0, f"terminator at {terminator:.2f}° incidence"
    assert night < 0.1 * (intercept + slope)          # the night side is dark


@pytest.mark.parametrize("viewport, dpr, phone", [
    ((1280, 900), 1, False),
    ((1280, 900), 2, False),
    ((2560, 1600), 2, False),
    ((390, 844), 3, True),          # iPhone class
    ((412, 915), 2.625, True),      # Android class
])
def test_colour_map_resolves_one_texel_per_pixel(webgl_browser, server, viewport, dpr, phone):
    """
    Every device loads the 4K map, and at the default zoom a texel covers at
    most one device pixel, the pixel ratio counted up to 2.  (An 8K map for
    3x phones would take ~180 MB of GPU memory, enough for iOS Safari to
    reload the tab.)
    """
    ctx = webgl_browser.new_context(viewport=dict(zip(("width", "height"), viewport)), device_scale_factor=dpr,
                                    is_mobile=phone, has_touch=phone)
    page = ctx.new_page()
    page.set_default_timeout(60_000)
    requested = []
    page.on("request", lambda r: requested.append(r.url) if "moon_color_" in r.url else None)
    page.route("**/static/textures/moon_*.jpg", lambda r: r.fulfill(status=200, content_type="image/png", body=FLAT_NORMAL))
    register(page, server["base"])
    wait_url(page, f"{server['base']}/")
    goto(page, f"{server['base']}/moon")
    try:
        page.wait_for_function("document.querySelector('#moon-container canvas')")
        page.wait_for_timeout(1000)
    except playwright_sync.TimeoutError:
        ctx.close()
        skip_unless_ci("the Moon page did not finish loading (error in moon.js, CDN unreachable or no WebGL)")
    canvas_height = page.evaluate("document.querySelector('#moon-container canvas').height")
    ctx.close()
    # Registering lands on the home page, which is also the Moon, so it may load twice
    assert requested and all("moon_color_4k" in url for url in requested), requested
    # surface per device pixel at distance 3, fov 45°, versus texel size
    counted_height = canvas_height * min(dpr, 2) / dpr
    surface_per_pixel = 2 * 2 * math.tan(math.radians(22.5)) / counted_height
    assert (2 * math.pi / 4096) / surface_per_pixel <= 1.0
