"""
Tests for Moon Viewer and Coordinate Conversion logic.
"""

import math
import pytest
from terrain.lunar_surface import (
    get_surface_provider,
    FallbackSurfaceProvider,
    set_surface_provider
)


# ---------------------------------------------------------------------------
# Coordinate Conversion logic matching frontend/moon.js
# ---------------------------------------------------------------------------

def js_local_to_lat_lon(x, y, z, radius=1.0):
    """
    Python equivalent of the frontend Three.js coordinate conversion.
    This validates our mathematical mapping between Three.js local space
    and Lunar latitude/longitude.
    """
    # Force y to be in bounds to prevent floating point errors with asin
    y_norm = max(-1.0, min(1.0, y / radius))
    lat_rad = math.asin(y_norm)
    lon_rad = math.atan2(x, z)
    return lat_rad * 180 / math.pi, lon_rad * 180 / math.pi


class TestCoordinateConversion:
    def test_north_pole(self):
        lat, lon = js_local_to_lat_lon(0, 1, 0)
        assert abs(lat - 90.0) < 1e-5

    def test_south_pole(self):
        lat, lon = js_local_to_lat_lon(0, -1, 0)
        assert abs(lat - (-90.0)) < 1e-5

    def test_equator_prime_meridian(self):
        # Prime meridian is at +Z
        lat, lon = js_local_to_lat_lon(0, 0, 1)
        assert abs(lat - 0.0) < 1e-5
        assert abs(lon - 0.0) < 1e-5

    def test_equator_90_east(self):
        # +X is 90 deg East
        lat, lon = js_local_to_lat_lon(1, 0, 0)
        assert abs(lat - 0.0) < 1e-5
        assert abs(lon - 90.0) < 1e-5

    def test_equator_90_west(self):
        # -X is 90 deg West
        lat, lon = js_local_to_lat_lon(-1, 0, 0)
        assert abs(lat - 0.0) < 1e-5
        assert abs(lon - (-90.0)) < 1e-5


# ---------------------------------------------------------------------------
# Lunar Surface Provider Tests
# ---------------------------------------------------------------------------

class TestFallbackSurfaceProvider:
    @pytest.fixture
    def provider(self):
        return FallbackSurfaceProvider(seed=42)

    def test_valid_coordinates_accepted(self, provider):
        # Should not raise
        provider.get_elevation(0, 0)
        provider.get_elevation(90, 180)
        provider.get_elevation(-90, -180)

    def test_invalid_latitude_rejected(self, provider):
        with pytest.raises(ValueError):
            provider.get_elevation(91, 0)
        with pytest.raises(ValueError):
            provider.get_elevation(-91, 0)

    def test_invalid_longitude_rejected(self, provider):
        with pytest.raises(ValueError):
            provider.get_elevation(0, 181)
        with pytest.raises(ValueError):
            provider.get_elevation(0, -181)

    def test_get_provider_name_explicitly_states_synthetic(self, provider):
        name = provider.get_provider_name()
        assert "NOT real" in name or "Synthetic" in name or "Fallback" in name
        assert provider.is_real_data() is False

    def test_elevation_bounds(self, provider):
        # We know bounds are between ELEV_MIN and ELEV_MAX
        el = provider.get_elevation(0, 0)
        assert provider.ELEV_MIN <= el <= provider.ELEV_MAX

    def test_terrain_texture(self, provider):
        tex = provider.get_terrain_texture(width=16, height=8)
        assert tex.shape == (8, 16)
        assert tex.dtype == 'uint8'
        # Contains some variation, not just 0
        assert tex.max() > 0


# ---------------------------------------------------------------------------
# API Endpoint Tests
# ---------------------------------------------------------------------------

def test_api_moon_texture(monkeypatch):
    import tempfile
    import os
    
    # We test the API endpoint using TestClient
    _temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    os.environ["LUNALAB_DB_PATH"] = _temp_db.name
    
    from fastapi.testclient import TestClient
    from backend.api import app
    from database.models import init_db
    
    init_db(_temp_db.name)
    with TestClient(app) as client:
        res = client.get("/api/moon/terrain-texture")
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/octet-stream"
        
        # Default size is 1024 * 512 = 524288 bytes
        assert len(res.content) == 1024 * 512
        
    os.unlink(_temp_db.name)


# ---------------------------------------------------------------------------
# Real LOLA DEM provider
# ---------------------------------------------------------------------------

class TestLolaSurfaceProvider:
    @pytest.fixture(scope="class")
    def lola(self):
        from terrain.lunar_surface import LolaSurfaceProvider
        return LolaSurfaceProvider()

    def test_is_real_data(self, lola):
        assert lola.is_real_data() is True
        assert "LOLA" in lola.get_provider_name()

    def test_maria_are_low_and_far_side_highlands_high(self, lola):
        # Maria sit ~2-3 km below the reference sphere; the far-side
        # highlands near Korolev reach ~+10 km; South Pole-Aitken is ~-7 km.
        assert -3500 < lola.get_elevation(28.0, 17.5) < -1500     # Mare Serenitatis
        assert -3500 < lola.get_elevation(33.0, -16.0) < -1500    # Mare Imbrium
        assert lola.get_elevation(5.4, -158.6) > 8000              # near Korolev
        assert lola.get_elevation(-69.3, -172.0) < -5000           # SPA basin

    def test_longitude_wraps_at_antimeridian(self, lola):
        assert lola.get_elevation(0, 180) == pytest.approx(lola.get_elevation(0, -180), abs=1.0)

    def test_invalid_coordinates_rejected(self, lola):
        with pytest.raises(ValueError):
            lola.get_elevation(95, 0)

    def test_texture_shape_and_range(self, lola):
        tex = lola.get_terrain_texture(64, 32)
        assert tex.shape == (32, 64) and tex.dtype == 'uint8'
        assert tex.min() < 40 and tex.max() > 200


def test_default_provider_is_lola():
    import terrain.lunar_surface as ls
    ls.set_surface_provider(None)
    assert ls.get_surface_provider().is_real_data() is True


def test_api_moon_elevation():
    from fastapi.testclient import TestClient
    from backend.api import app
    with TestClient(app) as client:
        res = client.get("/api/moon/elevation", params={"lat": 28.0, "lon": 17.5})
        assert res.status_code == 200
        body = res.json()
        assert body["is_real_data"] is True
        assert body["elevation"] < -1500
        assert client.get("/api/moon/elevation", params={"lat": 100, "lon": 0}).status_code == 422
