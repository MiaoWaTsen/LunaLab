"""
Tests for the Simulation Module.

These tests validate the crater scaling model independently,
without any web server, API, or database dependency.
"""

import pytest
import math
from simulation.crater_model import (
    ImpactParameters,
    SimulationResult,
    ValidationError,
    run_simulation,
    generate_crater_profile,
    validate_parameters,
    LUNAR_GRAVITY,
    LUNAR_TARGET_DENSITY,
    DEFAULT_IMPACTOR_DENSITY,
)


# ---------------------------------------------------------------------------
# Test fixtures — reusable parameter sets
# ---------------------------------------------------------------------------

@pytest.fixture
def default_params():
    """A standard set of parameters for testing."""
    return ImpactParameters(
        impactor_diameter=10.0,
        impact_velocity=20000.0,
        impact_angle=45.0,
    )


@pytest.fixture
def vertical_params():
    """90° (vertical) impact for testing angle independence."""
    return ImpactParameters(
        impactor_diameter=10.0,
        impact_velocity=20000.0,
        impact_angle=90.0,
    )


# ---------------------------------------------------------------------------
# 1. Determinism — same input → same result
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_result(self, default_params):
        """Identical inputs must produce identical outputs."""
        r1 = run_simulation(default_params)
        r2 = run_simulation(default_params)
        assert r1.crater_diameter == r2.crater_diameter
        assert r1.crater_depth == r2.crater_depth
        assert r1.rim_height == r2.rim_height
        assert r1.energy_joules == r2.energy_joules
        assert r1.model_name == r2.model_name

    def test_different_runs_are_not_random(self, default_params):
        """Running many times should still be deterministic."""
        results = [run_simulation(default_params).crater_diameter for _ in range(10)]
        assert len(set(results)) == 1


# ---------------------------------------------------------------------------
# 2. Validation — hard errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_diameter_zero_raises(self):
        p = ImpactParameters(impactor_diameter=0, impact_velocity=20000, impact_angle=45)
        with pytest.raises(ValidationError, match="impactor_diameter"):
            run_simulation(p)

    def test_diameter_negative_raises(self):
        p = ImpactParameters(impactor_diameter=-5, impact_velocity=20000, impact_angle=45)
        with pytest.raises(ValidationError, match="impactor_diameter"):
            run_simulation(p)

    def test_velocity_zero_raises(self):
        p = ImpactParameters(impactor_diameter=10, impact_velocity=0, impact_angle=45)
        with pytest.raises(ValidationError, match="impact_velocity"):
            run_simulation(p)

    def test_velocity_negative_raises(self):
        p = ImpactParameters(impactor_diameter=10, impact_velocity=-100, impact_angle=45)
        with pytest.raises(ValidationError, match="impact_velocity"):
            run_simulation(p)

    def test_angle_zero_raises(self):
        p = ImpactParameters(impactor_diameter=10, impact_velocity=20000, impact_angle=0)
        with pytest.raises(ValidationError, match="impact_angle"):
            run_simulation(p)

    def test_angle_negative_raises(self):
        p = ImpactParameters(impactor_diameter=10, impact_velocity=20000, impact_angle=-10)
        with pytest.raises(ValidationError, match="impact_angle"):
            run_simulation(p)

    def test_angle_over_90_raises(self):
        p = ImpactParameters(impactor_diameter=10, impact_velocity=20000, impact_angle=91)
        with pytest.raises(ValidationError, match="impact_angle"):
            run_simulation(p)

    def test_density_zero_raises(self):
        p = ImpactParameters(
            impactor_diameter=10, impact_velocity=20000, impact_angle=45,
            impactor_density=0,
        )
        with pytest.raises(ValidationError, match="impactor_density"):
            run_simulation(p)

    def test_target_density_zero_raises(self):
        p = ImpactParameters(
            impactor_diameter=10, impact_velocity=20000, impact_angle=45,
            target_density=0,
        )
        with pytest.raises(ValidationError, match="target_density"):
            run_simulation(p)

    def test_gravity_zero_raises(self):
        p = ImpactParameters(
            impactor_diameter=10, impact_velocity=20000, impact_angle=45,
            surface_gravity=0,
        )
        with pytest.raises(ValidationError, match="surface_gravity"):
            run_simulation(p)


# ---------------------------------------------------------------------------
# 3. Validation — soft warnings
# ---------------------------------------------------------------------------

class TestValidationWarnings:
    def test_large_diameter_warning(self):
        p = ImpactParameters(impactor_diameter=200_000, impact_velocity=20000, impact_angle=45)
        warnings = validate_parameters(p)
        assert any(w.field == "impactor_diameter" for w in warnings)

    def test_small_diameter_warning(self):
        p = ImpactParameters(impactor_diameter=0.001, impact_velocity=20000, impact_angle=45)
        warnings = validate_parameters(p)
        assert any(w.field == "impactor_diameter" for w in warnings)

    def test_high_velocity_warning(self):
        p = ImpactParameters(impactor_diameter=10, impact_velocity=200_000, impact_angle=45)
        warnings = validate_parameters(p)
        assert any(w.field == "impact_velocity" for w in warnings)

    def test_low_velocity_warning(self):
        p = ImpactParameters(impactor_diameter=10, impact_velocity=500, impact_angle=45)
        warnings = validate_parameters(p)
        assert any(w.field == "impact_velocity" for w in warnings)

    def test_very_oblique_angle_warning(self):
        p = ImpactParameters(impactor_diameter=10, impact_velocity=20000, impact_angle=5)
        warnings = validate_parameters(p)
        assert any(w.field == "impact_angle" for w in warnings)

    def test_no_warning_for_normal_params(self, default_params):
        warnings = validate_parameters(default_params)
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# 4. Physical sanity checks
# ---------------------------------------------------------------------------

class TestPhysicalSanity:
    def test_crater_larger_than_impactor(self, default_params):
        """Crater should be larger than the impactor."""
        r = run_simulation(default_params)
        assert r.crater_diameter > default_params.impactor_diameter

    def test_larger_impactor_larger_crater(self):
        """Bigger impactor → bigger crater (all else equal)."""
        small = run_simulation(ImpactParameters(
            impactor_diameter=5, impact_velocity=20000, impact_angle=45
        ))
        large = run_simulation(ImpactParameters(
            impactor_diameter=50, impact_velocity=20000, impact_angle=45
        ))
        assert large.crater_diameter > small.crater_diameter

    def test_faster_impact_larger_crater(self):
        """Higher velocity → bigger crater (all else equal)."""
        slow = run_simulation(ImpactParameters(
            impactor_diameter=10, impact_velocity=10000, impact_angle=45
        ))
        fast = run_simulation(ImpactParameters(
            impactor_diameter=10, impact_velocity=30000, impact_angle=45
        ))
        assert fast.crater_diameter > slow.crater_diameter

    def test_vertical_impact_largest(self):
        """Vertical (90°) impact should produce the largest crater."""
        oblique = run_simulation(ImpactParameters(
            impactor_diameter=10, impact_velocity=20000, impact_angle=30
        ))
        vertical = run_simulation(ImpactParameters(
            impactor_diameter=10, impact_velocity=20000, impact_angle=90
        ))
        assert vertical.crater_diameter > oblique.crater_diameter

    def test_depth_is_positive(self, default_params):
        r = run_simulation(default_params)
        assert r.crater_depth > 0

    def test_rim_height_is_positive(self, default_params):
        r = run_simulation(default_params)
        assert r.rim_height > 0

    def test_energy_is_positive(self, default_params):
        r = run_simulation(default_params)
        assert r.energy_joules > 0

    def test_ejecta_volume_is_positive(self, default_params):
        r = run_simulation(default_params)
        assert r.ejecta_volume > 0

    def test_model_name_is_set(self, default_params):
        r = run_simulation(default_params)
        assert r.model_name
        assert isinstance(r.model_name, str)
        assert len(r.model_name) > 0


# ---------------------------------------------------------------------------
# 5. Profile generation
# ---------------------------------------------------------------------------

class TestCraterProfile:
    def test_profile_has_correct_length(self):
        profile = generate_crater_profile(100, 20, 5, num_points=200)
        assert len(profile["x"]) == 200
        assert len(profile["y"]) == 200

    def test_profile_is_symmetric(self):
        profile = generate_crater_profile(100, 20, 5, num_points=201)
        mid = len(profile["y"]) // 2
        # Centre should be the deepest
        assert profile["y"][mid] == min(profile["y"])

    def test_profile_deepest_at_center(self):
        profile = generate_crater_profile(100, 20, 5)
        ys = profile["y"]
        mid = len(ys) // 2
        assert ys[mid] < 0  # below surface

    def test_profile_edges_at_zero(self):
        profile = generate_crater_profile(100, 20, 5)
        # First and last should be at y ≈ 0
        assert abs(profile["y"][0]) < 0.01
        assert abs(profile["y"][-1]) < 0.01
