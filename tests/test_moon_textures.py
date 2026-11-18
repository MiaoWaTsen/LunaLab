"""
The 3D Moon's textures (built by scripts/build_moon_textures.py).

The relief is checked against an independent dataset: the LOLA LDEM_4 grid
the backend serves elevations from (PDS release, not the SVS kit the
textures come from).  If the normal map were shifted, flipped or mirrored,
its slopes would stop lining up with that grid's.
"""

import math
import os

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXTURES = os.path.join(ROOT, "frontend", "textures")
LDEM = os.path.join(ROOT, "terrain", "data", "lola_ldem4_m.npy")

# The textures the page loaded before this change: colour 238,093 B + height 488,333 B
PREVIOUS_LOAD_BYTES = 238_093 + 488_333
MAX_LOAD_INCREASE_BYTES = 15_000_000

LANDMARKS = {
    "Copernicus": (9.62, -20.08),
    "Tycho": (-43.31, -11.36),
    "Kepler": (8.12, -38.01),
    "Aristarchus": (23.73, -47.49),
    "Plato": (51.62, -9.38),
    "Theophilus": (-11.45, 26.28),
}


def texture(name):
    return Image.open(os.path.join(TEXTURES, name))


@pytest.mark.parametrize("name, size", [
    ("moon_color_4k.jpg", (4096, 2048)),
    ("moon_normal_4k.jpg", (4096, 2048)),
])
def test_textures_are_full_equirectangular_maps(name, size):
    assert texture(name).size == size


def test_page_load_stays_within_budget():
    load = os.path.getsize(os.path.join(TEXTURES, "moon_color_4k.jpg")) + os.path.getsize(os.path.join(TEXTURES, "moon_normal_4k.jpg"))
    assert load - PREVIOUS_LOAD_BYTES <= MAX_LOAD_INCREASE_BYTES


def test_color_map_has_no_shading_baked_in():
    """
    Shading drawn into the colour map would not move with the real Sun.
    High-pass colour must not correlate with an east- or west-lit hillshade
    (the map this replaced scored |r| = 0.32).
    """
    color = np.asarray(texture("moon_color_4k.jpg").convert("L").resize((1440, 720), Image.BOX), float)
    dem = np.load(LDEM).astype(float)
    east_slope = np.roll(dem, -1, axis=1) - np.roll(dem, 1, axis=1)
    band = slice(120, 600)                       # ±60° latitude
    hp = lambda a: a - (np.roll(a, 3, 1) + np.roll(a, -3, 1) + np.roll(a, 3, 0) + np.roll(a, -3, 0)) / 4
    r = np.corrcoef(hp(color)[band].ravel(), hp(east_slope)[band].ravel())[0, 1]
    assert abs(r) < 0.1


def _normal_east_and_lola_slope():
    normal = np.asarray(texture("moon_normal_4k.jpg").resize((1440, 720), Image.BOX), float)
    east_from_map = -(normal[..., 0] - 127.5)        # red = -slope east
    dem = np.load(LDEM).astype(float)
    lat = np.radians(np.linspace(89.875, -89.875, 720))[:, None]
    cell = 1737.4e3 * math.radians(0.25)
    east_from_lola = (np.roll(dem, -1, axis=1) - np.roll(dem, 1, axis=1)) / (2 * cell * np.cos(lat))
    return east_from_map, east_from_lola


@pytest.mark.parametrize("landmark", LANDMARKS)
def test_relief_is_registered_to_lola_at_landmarks(landmark):
    """Within 8° of each landmark, the best match is within 0.25° (one grid cell) of no shift."""
    from_map, from_lola = _normal_east_and_lola_slope()
    lat, lon = LANDMARKS[landmark]
    r, c, half, search = int((90 - lat) * 4), int((lon + 180) * 4), 16, 4
    window = from_map[r - half:r + half, c - half:c + half]
    window = (window - window.mean()) / window.std()
    best = (-1.0, None)
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            ref = from_lola[r - half + dy:r + half + dy, c - half + dx:c + half + dx]
            score = float((window * (ref - ref.mean()) / ref.std()).mean())
            if score > best[0]:
                best = (score, (dy, dx))
    score, (dy, dx) = best
    assert score > 0.6, f"{landmark}: relief does not match LOLA (r = {score:.2f})"
    assert max(abs(dy), abs(dx)) * 0.25 <= 0.25, f"{landmark}: relief shifted by ({-dy * 0.25}°, {dx * 0.25}°)"


def test_relief_is_not_mirrored():
    """A mirrored or sign-flipped east axis would anti-correlate with LOLA."""
    from_map, from_lola = _normal_east_and_lola_slope()
    band = slice(120, 600)
    r = np.corrcoef(from_map[band].ravel(), from_lola[band].ravel())[0, 1]
    assert r > 0.6
