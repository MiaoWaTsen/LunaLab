"""
Build the crater-count data files in simulation/data/ from their sources.

    python scripts/build_crater_data.py ROBBINS_CSV CHANGE6_SCC GEOUNITS_SHP

ROBBINS_CSV  The Robbins (2018) lunar crater database (~1.3 million craters
             ≥ 1 km), as distributed with depths on Zenodo, CC BY 4.0:
             https://zenodo.org/records/21500453  (LunarCraterDepths_072226.csv)
CHANGE6_SCC  An expert crater count of the Chang'e-6 landing site, from the
             Craterstats sample data, BSD 3-Clause:
             https://github.com/ggmichael/craterstats/blob/main/src/craterstats/sample/CE-6%208-km%20vicinity.scc
GEOUNITS_SHP The GeoUnits shapefile of the Unified Geologic Map of the Moon,
             1:5M (Fortezzo, Spudis & Harrel 2020), USGS, public domain:
             https://asc-astropedia.s3.us-west-2.amazonaws.com/Moon/Geology/Unified_Geologic_Map_of_the_Moon_GIS_v2.zip
             (Lunar_GIS/Shapefiles/GeoUnits.shp).  Reading it needs pyshp
             (pip install pyshp), used only by this script.

Writes:
    simulation/data/landing_site_craters.json  craters within 150 km of each landing site,
                                               flagged by whether they lie on the site's
                                               mapped geological unit
    simulation/data/change6_count.json         the Chang'e-6 count: area, boundary, craters
"""

import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.surface_dating import LANDING_SITES, MOON_RADIUS_KM, great_circle_km  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulation", "data")
MAX_DISTANCE_KM = 150.0
RADII_KM = list(range(25, 155, 5))   # count-area radii the page offers
GRID_STEP_DEG = 0.03                 # ~0.9 km cells for the unit-area integral
OUTLINE_POINTS = 1500                # per site, for the map


# ---------------------------------------------------------------------------
# Geological units
# ---------------------------------------------------------------------------

def _shape_rings_deg(shape) -> list:
    """Polygon rings in (lon, lat) degrees; the map is equidistant cylindrical, metres."""
    metres_per_degree = MOON_RADIUS_KM * 1000.0 * math.pi / 180.0
    points = np.asarray(shape.points) / metres_per_degree
    parts = list(shape.parts) + [len(points)]
    return [points[a:b] for a, b in zip(parts, parts[1:])]


def points_in_polygon(lon, lat, rings, box=None) -> np.ndarray:
    """
    Even-odd ray casting, vectorised over the points.

    Even-odd handles holes (crater units cut out of a mare unit) without
    knowing which ring is which.  `box` = (lat_min, lat_max, lon_max) skips
    the edges that cannot cross a ray from points inside it.
    """
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    odd = np.zeros(lon.shape, bool)
    for ring in rings:
        x1, y1, x2, y2 = ring[:-1, 0], ring[:-1, 1], ring[1:, 0], ring[1:, 1]
        if box is not None:
            lat_min, lat_max, lon_max = box
            keep = (np.maximum(y1, y2) >= lat_min) & (np.minimum(y1, y2) <= lat_max) & (np.minimum(x1, x2) <= lon_max)
            x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
        for s in range(0, len(x1), 2000):
            a, b = x1[s:s + 2000, None], y1[s:s + 2000, None]
            c, d = x2[s:s + 2000, None], y2[s:s + 2000, None]
            crosses = ((b > lat) != (d > lat)) & (lon < a + (lat - b) * (c - a) / np.where(d == b, 1.0, d - b))
            odd ^= crosses.sum(axis=0) % 2 == 1
    return odd


def _distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2.0 * MOON_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def find_site_units(shp_path: str) -> dict:
    """For each landing site: the geological unit it sits on, its area within each radius, and an outline."""
    import shapefile   # pyshp; only this build step needs it

    reader = shapefile.Reader(shp_path)
    metres_per_degree = MOON_RADIUS_KM * 1000.0 * math.pi / 180.0
    units = {}
    for site in LANDING_SITES:
        x, y = site.longitude * metres_per_degree, site.latitude * metres_per_degree
        match = None
        for index, shape in enumerate(reader.iterShapes()):
            x0, y0, x1, y1 = shape.bbox
            if x0 <= x <= x1 and y0 <= y <= y1 and points_in_polygon([site.longitude], [site.latitude], _shape_rings_deg(shape))[0]:
                match = index
                break
        if match is None:
            raise ValueError(f"no geological unit contains {site.name}")
        shape, record = reader.shape(match), reader.record(match).as_dict()
        rings = _shape_rings_deg(shape)

        # Grid over the 150 km cap: which cells are on the unit, and how far out
        dlat = math.degrees(MAX_DISTANCE_KM / MOON_RADIUS_KM)
        dlon = dlat / math.cos(math.radians(min(89.0, abs(site.latitude) + dlat)))
        lats = np.arange(site.latitude - dlat, site.latitude + dlat, GRID_STEP_DEG) + GRID_STEP_DEG / 2
        lons = np.arange(site.longitude - dlon, site.longitude + dlon, GRID_STEP_DEG) + GRID_STEP_DEG / 2
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        distance = _distance_km(site.latitude, site.longitude, lat_grid, lon_grid)
        cap = distance <= MAX_DISTANCE_KM
        box = (site.latitude - dlat - 0.1, site.latitude + dlat + 0.1, site.longitude + dlon + 0.1)
        on_unit = np.zeros(cap.shape, bool)
        on_unit[cap] = points_in_polygon(lon_grid[cap], lat_grid[cap], rings, box)
        cell_km2 = (MOON_RADIUS_KM * math.radians(GRID_STEP_DEG)) ** 2 * np.cos(np.radians(lat_grid))

        # Outline: the unit's boundary vertices inside the cap, thinned for the map
        outline = []
        for ring in rings:
            near = _distance_km(site.latitude, site.longitude, ring[:, 1], ring[:, 0]) <= MAX_DISTANCE_KM * 1.05
            if near.any():
                outline.append(ring[near])
        total = sum(len(r) for r in outline)
        stride = max(1, total // OUTLINE_POINTS)
        outline = [[[round(float(lo), 4), round(float(la), 4)] for lo, la in r[::stride]] for r in outline if len(r) >= stride * 3]

        units[site.id] = {
            "rings": rings,
            "box": box,
            "code": record["FIRST_Unit"],
            "period": record["FIRST_Un_1"],
            "description": record["FIRST_Un_2"],
            "unit_area_km2": round(record["AREA_GEO"] / 1e6, 1),
            "area_by_radius_km2": {str(r): round(float((cell_km2 * (on_unit & (distance <= r))).sum()), 1) for r in RADII_KM},
            "outline": outline,
        }
    return units


def build_landing_sites(csv_path: str, units: dict) -> dict:
    craters = {site.id: [] for site in LANDING_SITES}
    # 150 km is under 5° of latitude; skip far rows before the trigonometry
    lat_window = math.degrees(MAX_DISTANCE_KM / MOON_RADIUS_KM) + 0.1
    with open(csv_path, newline="") as f:
        rows = csv.reader(f)
        header = next(rows)
        i_lat, i_lon, i_d = (header.index(c) for c in ("LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG"))
        for row in rows:
            lat = float(row[i_lat])
            lon = float(row[i_lon])
            lon = lon - 360.0 if lon > 180.0 else lon
            for site in LANDING_SITES:
                if abs(lat - site.latitude) > lat_window:
                    continue
                distance = great_circle_km(site.latitude, site.longitude, lat, lon)
                if distance <= MAX_DISTANCE_KM:
                    craters[site.id].append([round(lat, 4), round(lon, 4), round(float(row[i_d]), 4), round(distance, 2)])

    # Flag the craters whose centres lie on the site's unit
    for site in LANDING_SITES:
        rows = np.array(craters[site.id], float).reshape(-1, 4)
        unit = units[site.id]
        on_unit = points_in_polygon(rows[:, 1], rows[:, 0], unit["rings"], unit["box"])
        craters[site.id] = [row + [int(flag)] for row, flag in zip(craters[site.id], on_unit)]

    return {
        "source": "Robbins (2018) lunar crater database, via Zenodo record 21500453 (CC BY 4.0)",
        "unit_source": "Unified Geologic Map of the Moon, 1:5M (Fortezzo, Spudis & Harrel 2020), USGS",
        "max_distance_km": MAX_DISTANCE_KM,
        "columns": ["latitude", "longitude", "diameter_km", "distance_km", "on_unit"],
        "sites": craters,
        "units": {
            site_id: {k: v for k, v in unit.items() if k not in ("rings", "box")}
            for site_id, unit in units.items()
        },
    }


def build_change6(scc_path: str) -> dict:
    """Parse a Craterstats .scc file: the area, its boundary and the craters."""
    area = None
    boundary, craters = [], []
    section = None
    with open(scc_path) as f:
        for line in f:
            if line.startswith("Total_area"):
                area = float(line.split("=")[1].split("<")[0])
            elif line.startswith("unit_boundary"):
                section = "boundary"
            elif line.startswith("crater ="):
                section = "craters"
            elif line.startswith("}"):
                section = None
            elif section and not line.startswith("#"):
                parts = line.split()
                if section == "boundary" and len(parts) == 5:
                    boundary.append([round(float(parts[3]), 6), round(float(parts[4]), 6)])
                elif section == "craters" and len(parts) >= 4:
                    # diameter (km), fraction counted, lon, lat
                    craters.append([round(float(parts[0]), 5), round(float(parts[1]), 4),
                                    round(float(parts[2]), 6), round(float(parts[3]), 6)])
    if area is None or not craters:
        raise ValueError(f"{scc_path} is not a Craterstats crater count")
    return {
        "source": "Craterstats sample data, 'CE-6 8-km vicinity.scc' (G. Michael; BSD 3-Clause)",
        "area_km2": area,
        "boundary": boundary,
        "columns": ["diameter_km", "fraction", "longitude", "latitude"],
        "craters": craters,
    }


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    os.makedirs(OUT_DIR, exist_ok=True)
    units = find_site_units(sys.argv[3])
    sites = build_landing_sites(sys.argv[1], units)
    with open(os.path.join(OUT_DIR, "landing_site_craters.json"), "w") as f:
        json.dump(sites, f, separators=(",", ":"))
    change6 = build_change6(sys.argv[2])
    with open(os.path.join(OUT_DIR, "change6_count.json"), "w") as f:
        json.dump(change6, f, separators=(",", ":"))
    for site_id, unit in sites["units"].items():
        on_unit = sum(row[4] for row in sites["sites"][site_id])
        print(f"{site_id:9s} {unit['code']:5s} {unit['unit_area_km2']:>11,.0f} km² unit, "
              f"{unit['area_by_radius_km2']['150']:>8,.0f} km² within 150 km, {on_unit} craters on it")
    print("Chang'e-6:", len(change6["craters"]), "craters")


if __name__ == "__main__":
    main()
