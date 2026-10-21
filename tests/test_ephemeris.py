"""
Lunar ephemeris checked against the worked examples in Meeus,
*Astronomical Algorithms* (2nd ed.), and against known lunar phases.
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from simulation.lunar_ephemeris import moon_position, illumination_at_jde, moon_illumination

JDE_1992_04_12 = 2448724.5   # 1992 April 12, 0h TD — Meeus examples 47.a, 48.a, 53.a


def test_moon_position_matches_meeus_47a():
    lam, beta, dist = moon_position(JDE_1992_04_12)
    assert lam == pytest.approx(133.162655, abs=0.01)
    assert beta == pytest.approx(-3.229126, abs=0.01)
    assert dist == pytest.approx(368409.7, abs=50)    # truncated series


def test_libration_matches_meeus_53a():
    m = illumination_at_jde(JDE_1992_04_12)
    assert m.sub_earth_lon == pytest.approx(-1.206, abs=0.05)
    assert m.sub_earth_lat == pytest.approx(4.194, abs=0.05)


def test_sub_solar_point_matches_meeus_53a():
    m = illumination_at_jde(JDE_1992_04_12)
    assert m.sub_solar_lon == pytest.approx(67.89, abs=0.1)
    assert m.sub_solar_lat == pytest.approx(1.46, abs=0.1)


def test_illuminated_fraction_matches_meeus_48a():
    assert illumination_at_jde(JDE_1992_04_12).illuminated_fraction == pytest.approx(0.6786, abs=0.002)


def test_new_moon_during_2024_eclipse():
    m = moon_illumination(datetime(2024, 4, 8, 18, 21, tzinfo=timezone.utc))
    assert m.phase_name == "New Moon"
    assert m.illuminated_fraction < 0.001
    assert abs(m.sub_solar_lon) > 175          # Sun over the far side


def test_full_moon_2024_04_23():
    m = moon_illumination(datetime(2024, 4, 23, 23, 49, tzinfo=timezone.utc))
    assert m.phase_name == "Full Moon"
    assert m.illuminated_fraction > 0.999
    assert abs(m.sub_solar_lon) < 10           # Sun over the near side


def test_libration_stays_within_physical_limits():
    # Optical libration never exceeds ~7.9° in longitude and ~6.9° in latitude
    for day in range(0, 60, 3):
        m = illumination_at_jde(2460676.5 + day)
        assert abs(m.sub_earth_lon) < 8.2
        assert abs(m.sub_earth_lat) < 7.0


def test_illumination_api_is_public_and_accepts_time():
    from backend.api import app
    with TestClient(app) as anon:
        res = anon.get("/api/moon/illumination", params={"at": "2024-04-23T23:49:00Z"})
        assert res.status_code == 200
        body = res.json()
        assert body["phase_name"] == "Full Moon"
        assert set(body["sub_solar"]) == {"latitude", "longitude"}
        assert anon.get("/api/moon/illumination").status_code == 200
