"""The Moon itself: elevation, sunlight and libration, relief texture."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Response

from simulation.lunar_ephemeris import moon_illumination
from terrain.lunar_surface import get_surface_provider

router = APIRouter(prefix="/api/moon", tags=["moon"])


@router.get("/elevation")
def get_moon_elevation(lat: float, lon: float):
    """Surface elevation (m, relative to the 1737.4 km sphere) at a coordinate."""
    provider = get_surface_provider()
    try:
        elevation = provider.get_elevation(lat, lon)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "latitude": lat,
        "longitude": lon,
        "elevation": elevation,
        "source": provider.get_provider_name(),
        "is_real_data": provider.is_real_data(),
    }


@router.get("/illumination")
def get_moon_illumination(at: Optional[datetime] = None):
    """
    Where the Sun and Earth are as seen from the Moon (Meeus ephemeris):
    sub-solar point, sub-Earth point (libration), phase.  Defaults to now.
    """
    when = at or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    m = moon_illumination(when)
    return {
        "at": when.astimezone(timezone.utc).isoformat(),
        "sub_solar": {"latitude": round(m.sub_solar_lat, 3), "longitude": round(m.sub_solar_lon, 3)},
        "sub_earth": {"latitude": round(m.sub_earth_lat, 3), "longitude": round(m.sub_earth_lon, 3)},
        "phase_angle": round(m.phase_angle, 2),
        "illuminated_fraction": round(m.illuminated_fraction, 4),
        "phase_name": m.phase_name,
        "waxing": m.waxing,
    }


@router.get("/terrain-texture")
def get_moon_terrain_texture():
    """Return an 8-bit raw displacement map for 3D sphere visualization."""
    provider = get_surface_provider()
    # Fixed resolution 1024x512 for Three.js
    tex = provider.get_terrain_texture(1024, 512)
    return Response(content=tex.tobytes(), media_type="application/octet-stream")
