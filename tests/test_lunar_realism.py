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
    MODEL_NAME,
    REGOLITH_MODEL_NAME,
    ValidationError,
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

    def test_basin_threshold_scales_inversely_with_gravity(self):
        # Found by mutation testing: `* scale` -> `/ scale` survived, because
        # the basin threshold was only ever tested at lunar gravity.
        # At twice lunar gravity the 300 km threshold halves to 150 km.
        assert classify_crater(150_000, surface_gravity=3.24) == "basin"
        assert classify_crater(149_000, surface_gravity=3.24) == "complex"

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


class TestRegolithScaling:
    def test_default_material_is_rock(self):
        assert sim(10).model_name == MODEL_NAME

    def test_regolith_uses_its_own_model_name(self):
        assert sim(1, target_material="regolith").model_name == REGOLITH_MODEL_NAME

    def test_unknown_material_is_rejected(self):
        with pytest.raises(ValidationError):
            sim(1, target_material="cheese")

    def test_porous_regolith_makes_smaller_craters_than_rock(self):
        rock = sim(1, target_density=1500)
        regolith = sim(1, target_density=1500, target_material="regolith")
        assert regolith.crater_diameter < rock.crater_diameter

    def test_regolith_crater_grows_with_size_and_speed(self):
        diams = [sim(L, target_material="regolith").crater_diameter for L in (0.1, 0.5, 2.0)]
        speeds = [sim(1, velocity=v, target_material="regolith").crater_diameter for v in (5e3, 15e3, 40e3)]
        assert diams == sorted(diams) and len(set(diams)) == 3
        assert speeds == sorted(speeds) and len(set(speeds)) == 3

    def test_steeper_impact_makes_larger_regolith_crater(self):
        shallow = sim(1, angle=20, target_material="regolith")
        steep = sim(1, angle=90, target_material="regolith")
        assert steep.crater_diameter > shallow.crater_diameter

    def test_small_regolith_crater_has_no_bedrock_warning(self):
        warnings = sim(1, target_material="regolith").warnings
        assert not any(w["field"] == "target_material" for w in warnings)

    def test_crater_through_the_regolith_warns(self):
        warnings = sim(100, target_material="regolith").warnings
        assert any(w["field"] == "target_material" for w in warnings)

    # Reference values from a separate hand implementation of Holsapple's
    # formula, cross-checked against the model.  Found by mutation testing:
    # the real-impact bands in test_real_impacts.py are too loose to notice a
    # wrong exponent, and the strength term only matters for tiny impactors.
    @pytest.mark.parametrize("diameter, velocity, angle, impactor_density, expected", [
        (0.25725, 25_600.0, 56.0, 1800.0, 10.0298),   # gravity regime (2013 meteoroid)
        (0.5, 3_000.0, 60.0, 23.0, 2.5266),           # large density contrast
        (0.001, 3_000.0, 90.0, 7800.0, 0.0501),       # strength regime: 1 mm iron grain
    ])
    def test_regolith_reference_values(self, diameter, velocity, angle, impactor_density, expected):
        result = sim(diameter, velocity=velocity, angle=angle,
                     impactor_density=impactor_density, target_density=1500,
                     target_material="regolith")
        assert result.crater_diameter == pytest.approx(expected, rel=2e-3)

    def test_strength_regime_crater_barely_depends_on_gravity(self):
        # A 1 mm grain is stopped by the regolith's cohesion, not by gravity:
        # six times the gravity shrinks it ~5 %, against ~26 % in the gravity regime.
        def grain(g):
            return sim(0.001, velocity=3000, angle=90, impactor_density=7800,
                       target_density=1500, surface_gravity=g,
                       target_material="regolith").crater_diameter
        assert grain(9.81) / grain(1.62) > 0.9


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
