"""
Lunar Surface Provider — terrain/lunar_surface.py

This module defines an abstract interface for lunar surface elevation data
and provides a fallback procedural generator.

SCIENTIFIC DISCLAIMER
======================
The ``FallbackSurfaceProvider`` does NOT use real lunar elevation data.
It generates a procedural height field using layered noise that *loosely*
resembles lunar topography for visualization purposes only.

When a real Lunar DEM (e.g. SLDEM2015 — Barker et al. 2016) becomes
available, implement ``SLDEM2015Provider`` using the same interface so
that the rest of the system (API, frontend, simulation) remains unchanged.

Provider Interface
-------------------
Any provider must implement:
    - get_elevation(lat, lon) → float  (metres)
    - get_terrain_texture(width, height) → numpy array (uint8 grayscale)
    - get_provider_name() → str
    - is_real_data() → bool

References
-----------
    - SLDEM2015: Barker, M. K. et al. (2016). "A new lunar digital
      elevation model from the Lunar Orbiter Laser Altimeter and SELENE
      Terrain Camera," Icarus, 273, 346–355.
      License: NASA public domain.
    - Data source: http://imbrium.mit.edu/DATA/SLDEM2015/
"""

import math
import hashlib
import struct
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class LunarSurfaceProvider(ABC):
    """
    Abstract interface for lunar surface elevation data.

    All providers must return elevation in metres relative to a reference
    sphere (mean lunar radius ≈ 1737.4 km).
    """

    @abstractmethod
    def get_elevation(self, latitude: float, longitude: float) -> float:
        """
        Return surface elevation at a given coordinate.

        Parameters
        ----------
        latitude : float
            Degrees, -90 to +90.
        longitude : float
            Degrees, -180 to +180.

        Returns
        -------
        float
            Elevation in metres relative to the reference sphere.
        """

    @abstractmethod
    def get_terrain_texture(
        self, width: int = 2048, height: int = 1024
    ) -> np.ndarray:
        """
        Return a grayscale displacement map for 3D sphere rendering.

        The map uses equirectangular projection:
            - x axis: longitude (-180 to +180)
            - y axis: latitude  (-90 to +90)

        Parameters
        ----------
        width : int
            Texture width in pixels.
        height : int
            Texture height in pixels.

        Returns
        -------
        np.ndarray
            uint8 grayscale array of shape (height, width).
            0 = lowest elevation, 255 = highest elevation.
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a human-readable name for this provider."""

    @abstractmethod
    def is_real_data(self) -> bool:
        """Return True if this provider uses real Lunar DEM data."""

    def validate_coordinates(self, latitude: float, longitude: float) -> None:
        """Raise ValueError if coordinates are out of range."""
        if latitude < -90 or latitude > 90:
            raise ValueError(
                f"latitude must be in [-90, 90], got {latitude}"
            )
        if longitude < -180 or longitude > 180:
            raise ValueError(
                f"longitude must be in [-180, 180], got {longitude}"
            )


# ---------------------------------------------------------------------------
# Deterministic hash-based noise (no external dependency)
# ---------------------------------------------------------------------------

def _hash_noise_2d(x: float, y: float, seed: int = 0) -> float:
    """
    Deterministic pseudo-noise function using SHA-256 hashing.

    Returns a value in [0, 1) for any (x, y) input.
    This is NOT cryptographic — it's used for procedural terrain generation.
    """
    data = struct.pack('<ddi', x, y, seed)
    h = hashlib.sha256(data).digest()
    # Take first 4 bytes as uint32, normalise to [0, 1)
    val = struct.unpack('<I', h[:4])[0]
    return val / 4294967296.0


def _smooth_noise(x: float, y: float, seed: int = 0) -> float:
    """Bilinear-interpolated lattice noise."""
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    fx = x - ix
    fy = y - iy

    # Smoothstep interpolation
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)

    n00 = _hash_noise_2d(ix, iy, seed)
    n10 = _hash_noise_2d(ix + 1, iy, seed)
    n01 = _hash_noise_2d(ix, iy + 1, seed)
    n11 = _hash_noise_2d(ix + 1, iy + 1, seed)

    nx0 = n00 + fx * (n10 - n00)
    nx1 = n01 + fx * (n11 - n01)

    return nx0 + fy * (nx1 - nx0)


def _fractal_noise(
    x: float, y: float, octaves: int = 6, seed: int = 0
) -> float:
    """
    Multi-octave fractal noise (fBm-like).

    Returns a value roughly in [0, 1].
    """
    value = 0.0
    amplitude = 0.5
    frequency = 1.0
    max_val = 0.0

    for i in range(octaves):
        value += amplitude * _smooth_noise(
            x * frequency, y * frequency, seed + i * 137
        )
        max_val += amplitude
        amplitude *= 0.5
        frequency *= 2.0

    return value / max_val if max_val > 0 else 0.0


# ---------------------------------------------------------------------------
# Known large lunar features (approximate centres and radii)
# Used by fallback provider to add plausible crater-like depressions.
# Coordinates: (lat°, lon°, radius_deg, depth_km)
# ---------------------------------------------------------------------------

_MAJOR_FEATURES = [
    # Mare / large basins (broad shallow depressions)
    (26.0, 3.0, 12.0, 2.5),      # Mare Serenitatis (approx)
    (15.0, -2.0, 10.0, 2.0),     # Mare Imbrium region
    (-15.0, -20.0, 8.0, 1.8),    # Mare Humorum
    (7.0, 22.0, 8.0, 1.5),       # Mare Tranquillitatis
    (-10.0, -40.0, 7.0, 1.5),    # Oceanus Procellarum edge
    (47.0, 0.0, 6.0, 2.0),       # Near Plato
    # Large craters
    (-43.0, -11.0, 5.0, 3.5),    # Tycho
    (9.6, 20.0, 4.5, 3.0),       # Near Copernicus
    (-8.9, -58.0, 5.5, 3.0),     # Kepler region
]


# ---------------------------------------------------------------------------
# Fallback provider — procedural noise
# ---------------------------------------------------------------------------

class FallbackSurfaceProvider(LunarSurfaceProvider):
    """
    Procedural lunar surface approximation.

    Generates a height field using layered noise with crater-like
    depressions at known mare / large crater locations.

    IMPORTANT: This is NOT real lunar data.  It is a synthetic
    approximation for visualization when no DEM file is available.
    """

    # Elevation range (metres) — roughly matching real Moon
    # Real Moon: roughly -9 km to +11 km from mean radius
    ELEV_MIN = -8500.0   # metres
    ELEV_MAX = 10700.0   # metres

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._texture_cache: Optional[np.ndarray] = None
        self._texture_cache_key: Optional[tuple] = None

    def get_elevation(self, latitude: float, longitude: float) -> float:
        """Return procedural elevation at (lat, lon) in metres."""
        self.validate_coordinates(latitude, longitude)
        raw = self._raw_elevation(latitude, longitude)
        return round(raw, 2)

    def get_terrain_texture(
        self, width: int = 2048, height: int = 1024
    ) -> np.ndarray:
        """
        Return a grayscale displacement texture.

        Uses vectorised NumPy for performance (~200ms for 2048×1024).
        Result is cached for repeated calls with the same dimensions.
        """
        cache_key = (width, height)
        if (
            self._texture_cache is not None
            and self._texture_cache_key == cache_key
        ):
            return self._texture_cache

        # Build coordinate grids
        lons = np.linspace(-180, 180, width)
        lats = np.linspace(90, -90, height)  # top=north
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Vectorised elevation computation
        elev = self._raw_elevation_vectorised(lat_grid, lon_grid)

        # Normalise to [0, 255]
        e_min = self.ELEV_MIN
        e_max = self.ELEV_MAX
        normalised = np.clip((elev - e_min) / (e_max - e_min), 0, 1)
        texture = (normalised * 255).astype(np.uint8)

        self._texture_cache = texture
        self._texture_cache_key = cache_key
        return texture

    def get_provider_name(self) -> str:
        return "Fallback Procedural Surface (NOT real Lunar DEM)"

    def is_real_data(self) -> bool:
        return False

    # ── Internal methods ──────────────────────────────────────────────

    def _raw_elevation(self, lat: float, lon: float) -> float:
        """Compute elevation for a single point (scalar)."""
        # Convert to normalised coordinates for noise
        nx = (lon + 180) / 360.0 * 20.0
        ny = (lat + 90) / 180.0 * 10.0

        # Base terrain from fractal noise
        base = _fractal_noise(nx, ny, octaves=6, seed=self._seed)

        # Scale to elevation range
        elev = self.ELEV_MIN + base * (self.ELEV_MAX - self.ELEV_MIN)

        # Add crater depressions for major features
        for feat_lat, feat_lon, feat_r, feat_depth in _MAJOR_FEATURES:
            dist = math.sqrt(
                (lat - feat_lat) ** 2 + (lon - feat_lon) ** 2
            )
            if dist < feat_r * 1.5:
                # Gaussian depression
                sigma = feat_r * 0.6
                depression = feat_depth * 1000 * math.exp(
                    -0.5 * (dist / sigma) ** 2
                )
                elev -= depression

        return elev

    def _raw_elevation_vectorised(
        self, lat_grid: np.ndarray, lon_grid: np.ndarray
    ) -> np.ndarray:
        """
        Vectorised elevation computation for texture generation.

        Uses a simplified noise approximation for performance:
        multiple sine/cosine harmonics instead of per-pixel hash noise.
        This is faster but slightly different from the scalar version.
        """
        # Normalised coordinates
        nx = (lon_grid + 180) / 360.0
        ny = (lat_grid + 90) / 180.0

        # Multi-harmonic approximation (deterministic, vectorised)
        elev = np.zeros_like(nx, dtype=np.float64)
        amplitude = 0.5
        for k in range(1, 8):
            freq = k * 2.7 + self._seed * 0.01
            elev += amplitude * np.sin(
                nx * freq * 2 * np.pi + k * 1.3
            ) * np.cos(
                ny * freq * np.pi + k * 0.7
            )
            amplitude *= 0.55

        # Normalise to [0, 1]
        elev = (elev - elev.min()) / (elev.max() - elev.min() + 1e-10)

        # Scale to elevation range
        elev = self.ELEV_MIN + elev * (self.ELEV_MAX - self.ELEV_MIN)

        # Add major feature depressions
        for feat_lat, feat_lon, feat_r, feat_depth in _MAJOR_FEATURES:
            dist = np.sqrt(
                (lat_grid - feat_lat) ** 2 + (lon_grid - feat_lon) ** 2
            )
            sigma = feat_r * 0.6
            depression = feat_depth * 1000 * np.exp(
                -0.5 * (dist / sigma) ** 2
            )
            elev -= depression

        return elev


# ---------------------------------------------------------------------------
# Module-level default provider instance
# ---------------------------------------------------------------------------

_default_provider: Optional[LunarSurfaceProvider] = None


def get_surface_provider() -> LunarSurfaceProvider:
    """
    Return the current lunar surface provider.

    Uses FallbackSurfaceProvider by default.  To switch to a real DEM
    provider, call ``set_surface_provider()`` at application startup.
    """
    global _default_provider
    if _default_provider is None:
        _default_provider = FallbackSurfaceProvider()
    return _default_provider


def set_surface_provider(provider: LunarSurfaceProvider) -> None:
    """Replace the global surface provider (e.g. with a real DEM loader)."""
    global _default_provider
    _default_provider = provider
