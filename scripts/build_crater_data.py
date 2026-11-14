"""
Build the crater-count data files in simulation/data/ from their sources.

    python scripts/build_crater_data.py ROBBINS_CSV CHANGE6_SCC

ROBBINS_CSV  The Robbins (2018) lunar crater database (~1.3 million craters
             ≥ 1 km), as distributed with depths on Zenodo, CC BY 4.0:
             https://zenodo.org/records/21500453  (LunarCraterDepths_072226.csv)
CHANGE6_SCC  An expert crater count of the Chang'e-6 landing site, from the
             Craterstats sample data, BSD 3-Clause:
             https://github.com/ggmichael/craterstats/blob/main/src/craterstats/sample/CE-6%208-km%20vicinity.scc

Writes:
    simulation/data/landing_site_craters.json  craters within 150 km of each landing site
    simulation/data/change6_count.json         the Chang'e-6 count: area, boundary, craters
"""

import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.surface_dating import LANDING_SITES, MOON_RADIUS_KM, great_circle_km  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulation", "data")
MAX_DISTANCE_KM = 150.0


def build_landing_sites(csv_path: str) -> dict:
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
    return {
        "source": "Robbins (2018) lunar crater database, via Zenodo record 21500453 (CC BY 4.0)",
        "max_distance_km": MAX_DISTANCE_KM,
        "columns": ["latitude", "longitude", "diameter_km", "distance_km"],
        "sites": craters,
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
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    os.makedirs(OUT_DIR, exist_ok=True)
    sites = build_landing_sites(sys.argv[1])
    with open(os.path.join(OUT_DIR, "landing_site_craters.json"), "w") as f:
        json.dump(sites, f, separators=(",", ":"))
    change6 = build_change6(sys.argv[2])
    with open(os.path.join(OUT_DIR, "change6_count.json"), "w") as f:
        json.dump(change6, f, separators=(",", ":"))
    print({k: len(v) for k, v in sites["sites"].items()}, "| Chang'e-6:", len(change6["craters"]), "craters")


if __name__ == "__main__":
    main()
