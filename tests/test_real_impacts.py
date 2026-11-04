"""
Validation against real lunar impacts with a measured crater.

The other model tests check physical properties (monotonic, scales with
gravity) because there is no exact expected value to compare against.  These
two impacts are the exception: the impactor was measured, and the Lunar
Reconnaissance Orbiter later photographed the crater.

The default "rock" scaling overestimates both craters by about 2x: its
exponents (μ ≈ 0.55) describe a competent, non-porous target, but the Moon's
surface is porous regolith that absorbs energy as it compacts.  The
"regolith" option (Holsapple 1993 scaling with μ = 0.4) brings both closer.

The tests pin each bias to a band, so a change that makes it worse (or a
fix that makes it better) shows up here and the documented numbers get
updated.

Both use a regolith target density of 1500 kg/m³, as Moser et al. assumed.
"""

import math

import pytest

from simulation.crater_model import ImpactParameters, run_simulation

REGOLITH_DENSITY = 1500.0  # kg/m³


def sphere_diameter(mass: float, density: float) -> float:
    return (6.0 * mass / (math.pi * density)) ** (1.0 / 3.0)


# 2013-03-17 meteoroid impact (Moser, Suggs & Suggs 2014, Asteroids, Comets,
# Meteors, Helsinki): 16 kg at 25.6 km/s, 56° from horizontal, impactor
# density 1800-3000 kg/m³.  LRO found an 18 m rim-to-rim crater.
METEOROID_MASS = 16.0
METEOROID_CRATER = 18.0


def meteoroid(impactor_density: float, material: str):
    return run_simulation(ImpactParameters(
        impactor_diameter=sphere_diameter(METEOROID_MASS, impactor_density),
        impact_velocity=25_600.0,
        impact_angle=56.0,
        impactor_density=impactor_density,
        target_density=REGOLITH_DENSITY,
        target_material=material,
    ))


# Apollo 14 S-IVB rocket stage (LROC, Robinson 2009): 14,016 kg at
# 2.54 km/s, 69° from horizontal; crater about 35 m across.  The stage is a
# hollow 17.8 m x 6.6 m cylinder, modelled as a sphere of the same volume
# and mass, so this case is rougher than the meteoroid.
S_IVB_MASS = 14_016.0
S_IVB_VOLUME = math.pi * 3.3 ** 2 * 17.8
S_IVB_CRATER = 35.0


def s_ivb(material: str):
    density = S_IVB_MASS / S_IVB_VOLUME
    return run_simulation(ImpactParameters(
        impactor_diameter=sphere_diameter(S_IVB_MASS, density),
        impact_velocity=2_540.0,
        impact_angle=69.0,
        impactor_density=density,
        target_density=REGOLITH_DENSITY,
        target_material=material,
    ))


@pytest.mark.parametrize("impactor_density", [1800.0, 3000.0])
def test_2013_meteoroid_rock_scaling_overestimates_about_2x(impactor_density):
    result = meteoroid(impactor_density, "rock")
    # The paper's kinetic energy, 5.4e9 J, checks the inputs themselves
    assert result.energy_joules == pytest.approx(5.4e9, rel=0.05)
    assert 1.5 < result.crater_diameter / METEOROID_CRATER < 2.5


@pytest.mark.parametrize("impactor_density", [1800.0, 3000.0])
def test_2013_meteoroid_regolith_scaling_underestimates(impactor_density):
    # 10 m, the low end of the 10-23 m Moser et al. predicted with the same
    # Holsapple scaling.  The rest of the gap is within the uncertainty of
    # the 16 kg mass, which comes from an assumed luminous efficiency.
    result = meteoroid(impactor_density, "regolith")
    assert 0.45 < result.crater_diameter / METEOROID_CRATER < 0.7


def test_apollo14_s_ivb_rock_scaling_overestimates_about_2x():
    assert 1.5 < s_ivb("rock").crater_diameter / S_IVB_CRATER < 2.5


def test_apollo14_s_ivb_regolith_scaling_within_20_percent():
    assert 0.8 < s_ivb("regolith").crater_diameter / S_IVB_CRATER < 1.2


@pytest.mark.parametrize("case", ["meteoroid", "s_ivb"])
def test_regolith_scaling_is_closer_to_observation_than_rock(case):
    def error(material):
        if case == "meteoroid":
            predicted, observed = meteoroid(1800.0, material).crater_diameter, METEOROID_CRATER
        else:
            predicted, observed = s_ivb(material).crater_diameter, S_IVB_CRATER
        return abs(math.log(predicted / observed))   # 2x too big == 2x too small

    assert error("regolith") < error("rock")
