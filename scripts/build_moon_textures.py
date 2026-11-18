"""
Build the 3D Moon's textures in frontend/textures/ from the NASA SVS CGI Moon Kit.

    python scripts/build_moon_textures.py COLOR_8K_TIF LDEM_16_TIF

COLOR_8K_TIF  lroc_color_poles_8k.tif (8192x4096, 24-bit RGB), the LROC WAC colour
              mosaic with the polar gaps filled, centred on 0° longitude:
              https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_poles_8k.tif
LDEM_16_TIF   ldem_16.tif (5760x2880, float, km above the 1737.4 km sphere), LOLA
              elevation at 16 px/deg:
              https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_16.tif

Both come from the CGI Moon Kit, NASA's Scientific Visualization Studio
(https://svs.gsfc.nasa.gov/4720), NASA imagery free of copyright.

Writes:
    frontend/textures/moon_color_4k.jpg   colour, 4096x2048
    frontend/textures/moon_normal_4k.jpg  tangent-space normal map from LOLA slopes

The colour map has no shading baked in (unlike the map it replaces), so all
relief shading comes from the normal map and moves with the real Sun.
Resampling uses whole-pixel box filters on the same equirectangular grid, so
no feature moves: pixel (i, j) of every output covers the same lat/lon cell.
"""

import math
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None   # the 8K map is 33 Mpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "frontend", "textures")
MOON_RADIUS_KM = 1737.4
JPEG_QUALITY = 90
NORMAL_SIZE = (4096, 2048)
# True slopes: the Moon's relief at ~2 km scale is gentle, and shows as on the
# real Moon, strongest near the terminator.
RELIEF_EXAGGERATION = 1.0


def slopes(elevation_km: np.ndarray):
    """East and north slopes (dimensionless) of an equirectangular grid whose row 0 is the north edge."""
    rows, cols = elevation_km.shape
    cell_km = MOON_RADIUS_KM * math.radians(360.0 / cols)
    lat = np.radians(90.0 - (np.arange(rows) + 0.5) * 180.0 / rows)
    # Toward the poles a column shrinks to cos(lat) of a row, and a one-column
    # difference turns the grid's noise into radial streaks.  Difference over
    # 1/cos(lat) columns instead, so every slope spans ~1 row (~1.9 km).
    east = np.empty_like(elevation_km)
    for r in range(rows):
        step = int(min(cols // 4, max(1, round(1 / math.cos(lat[r])))))
        row = elevation_km[r]
        east[r] = (np.roll(row, -step) - np.roll(row, step)) / (2 * step * cell_km * math.cos(lat[r]))
    padded = np.pad(elevation_km, ((1, 1), (0, 0)), mode="edge")
    north = (padded[:-2] - padded[2:]) / (2 * cell_km)
    return east, north


def normal_map(elevation_km: np.ndarray, size, exaggeration: float = RELIEF_EXAGGERATION) -> Image.Image:
    """
    Tangent-space normal map: red = east, green = north, blue = up.

    three.js reads red along +u and green along +v; on the sphere u grows
    eastward and v northward (the texture is flipped on upload), so these
    are the surface's own east and north.
    """
    east, north = slopes(elevation_km)
    east = np.asarray(Image.fromarray(east.astype(np.float32), mode="F").resize(size, Image.BOX))
    north = np.asarray(Image.fromarray(north.astype(np.float32), mode="F").resize(size, Image.BOX))
    n = np.stack([-exaggeration * east, -exaggeration * north, np.ones_like(east)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return Image.fromarray(np.round((n + 1.0) * 127.5).clip(0, 255).astype(np.uint8), mode="RGB")


def save_jpeg(image: Image.Image, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    image.save(path, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
    print(f"{name:22s} {image.size[0]}x{image.size[1]}  {os.path.getsize(path) / 1e6:.2f} MB")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    color = Image.open(sys.argv[1]).convert("RGB")
    if color.size != (8192, 4096):
        raise ValueError(f"expected the 8192x4096 colour map, got {color.size}")
    elevation = np.asarray(Image.open(sys.argv[2]), dtype=np.float64)
    if elevation.shape != (2880, 5760):
        raise ValueError(f"expected the 5760x2880 LOLA grid, got {elevation.shape[::-1]}")

    os.makedirs(OUT_DIR, exist_ok=True)
    save_jpeg(color.resize((4096, 2048), Image.BOX), "moon_color_4k.jpg")
    save_jpeg(normal_map(elevation, NORMAL_SIZE), "moon_normal_4k.jpg")


if __name__ == "__main__":
    main()
