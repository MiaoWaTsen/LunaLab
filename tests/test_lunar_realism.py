"""
Realism checks for the Collins-Pike lunar scaling model and terrain shapes.

These pin the model to well-known reference behaviour rather than to exact
numbers, so coefficients can be refined without rewriting the tests.
"""

import math
import numpy as np
import pytest

from simulation.crater_model import (
    ImpactParameters,
    run_simulation,
    classify_crater,
    transition_diameter,
    LUNAR_TRANSITION_DIAMETER,
)
from terrain.crater_terrain import radial_elevation
from terrain.asymmetric_crater import ejecta_weight, axis_ratio, generate_asymmetric_elevation_field


def sim(diameter, velocity=18000.0, angle=45.0, **kw):
    return run_simulation(ImpactParameters(diameter, velocity, angle, **kw))


class TestCraterRegimes:
    def test_small_crater_is_simple_with_depth_one_fifth(self):
        r = sim(10)
        assert r.crater_type == "simple"
        assert r.crater_depth / r.crater_diameter == pytest.approx(0.2, rel=1e-3)

    def test_tycho_scale_impact_is_complex_and_tycho_sized(self):
        # A ~6 km asteroid at 16 km/s should give a Tycho-like (85 km, ~4.8 km deep) crater
        r = sim(6000, velocity=16000)
        assert r.crater_type == "complex"
        assert 70_000 < r.crater_diameter < 100_000
        assert 4_000 < r.crater_depth < 6_000

    def test_complex_craters_are_shallower_relative_to_diameter(self):
        simple, complex_ = sim(100), sim(6000)
        assert complex_.crater_depth / complex_.crater_diameter < simple.crater_depth / simple.crater_diameter

    def test_basin_classification(self):
        assert classify_crater(500_000) == "basin"
        assert classify_crater(50_000) == "complex"
        assert classify_crater(5_000) == "simple"

    def test_transition_diameter_scales_inversely_with_gravity(self):
        assert transition_diameter(1.62) == pytest.approx(LUNAR_TRANSITION_DIAMETER)
        assert transition_diameter(3.24) == pytest.approx(LUNAR_TRANSITION_DIAMETER / 2)

    def test_diameter_and_depth_monotonic_across_transition(self):
        sizes = np.geomspace(200, 20_000, 80)
        results = [sim(float(L)) for L in sizes]
        diams = [r.crater_diameter for r in results]
        depths = [r.crater_depth for r in results]
        assert all(b > a for a, b in zip(diams, diams[1:]))
        assert all(b >= a for a, b in zip(depths, depths[1:]))

    def test_lower_gravity_makes_larger_crater(self):
        moon = sim(50, surface_gravity=1.62)
        earth = sim(50, surface_gravity=9.81)
        assert moon.crater_diameter > earth.crater_diameter


class TestVelocityWarnings:
    def test_below_lunar_escape_velocity_warns(self):
        assert any(w["field"] == "impact_velocity" for w in sim(10, velocity=2000).warnings)

    def test_typical_lunar_velocity_has_no_warning(self):
        assert not any(w["field"] == "impact_velocity" for w in sim(10, velocity=18000).warnings)

    def test_above_solar_system_limit_warns(self):
        assert any(w["field"] == "impact_velocity" for w in sim(10, velocity=80000).warnings)


class TestRadialShape:
    @pytest.mark.parametrize("ctype", ["simple", "complex", "basin"])
    def test_profile_is_continuous_at_rim(self, ctype):
        inner = radial_elevation(np.array([1.0 - 1e-6]), 100.0, 20.0, ctype)[0]
        outer = radial_elevation(np.array([1.0 + 1e-6]), 100.0, 20.0, ctype)[0]
        assert inner == pytest.approx(20.0, abs=0.01)
        assert outer == pytest.approx(20.0, abs=0.01)

    def test_complex_crater_has_central_peak(self):
        z = radial_elevation(np.array([0.0, 0.3]), 100.0, 20.0, "complex")
        assert z[0] > z[1]  # peak rises above the flat floor

    def test_complex_crater_has_flat_floor(self):
        z = radial_elevation(np.array([0.3, 0.38]), 100.0, 20.0, "complex")
        assert z[0] == pytest.approx(z[1], abs=1.0)


class TestObliqueImpacts:
    phi = np.linspace(0, 2 * math.pi, 360, endpoint=False)

    def test_steep_impact_has_symmetric_ejecta(self):
        w = ejecta_weight(self.phi, 60.0)
        assert np.allclose(w, 1.0)

    def test_ejecta_weight_preserves_volume(self):
        for angle in (5, 15, 30, 44):
            assert ejecta_weight(self.phi, angle).mean() == pytest.approx(1.0, rel=1e-2)

    def test_forbidden_zone_uprange_below_25_degrees(self):
        w = ejecta_weight(np.array([0.0, math.pi]), 20.0)
        assert w[1] == pytest.approx(0.0, abs=1e-9)  # uprange
        assert w[0] > 1.0                              # downrange

    def test_butterfly_pattern_below_10_degrees(self):
        w = ejecta_weight(np.array([0.0, math.pi / 2, math.pi]), 2.0)
        assert w[1] > w[0] and w[1] > w[2]  # cross-range dominates

    def test_crater_only_elongates_below_15_degrees(self):
        assert axis_ratio(30.0) == 1.0
        assert axis_ratio(15.0) == 1.0
        assert axis_ratio(5.0) < 1.0

    def test_vertical_field_matches_steep_oblique_field(self):
        a = generate_asymmetric_elevation_field(1000, 200, 36, impact_angle=90, width=64, height=32)
        b = generate_asymmetric_elevation_field(1000, 200, 36, impact_angle=50, width=64, height=32)
        assert a["z"] == b["z"]
        assert b["is_asymmetric"] is False
