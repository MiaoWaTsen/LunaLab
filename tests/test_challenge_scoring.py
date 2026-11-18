"""
Boundary tests for challenge scoring (simulation/challenges.py).

Grades are cut at 90 (S), 75 (A) and 60 (B) points, and the grade decides
the XP awarded, so each cut is tested exactly on the line and just below it
(boundary value analysis).

Scores are driven through the Tycho mission, whose score is
    0.6 * diameter_score + 0.4 * depth_score
With the diameter exact (diameter_score = 100) and a depth error of e %,
the score is 100 - 0.8 e.  So a depth error of 12.5 % and 31.25 % lands
exactly on 90 and 75 points.  60 points needs both errors: 20 % on the
diameter (score 50) and 12.5 % on the depth (score 75) give 30 + 30.

A measurement that scores zero (diameter >= 40 % off, depth >= 50 % off)
fails outright, whatever the total.
"""

import pytest

from simulation.challenges import GRADE_XP, award_challenge_xp, evaluate_challenge

TYCHO_DIAMETER = 85_000.0
TYCHO_DEPTH = 4_800.0


def tycho(depth_error_pct: float, diameter_error_pct: float = 0.0) -> dict:
    """Evaluate a Tycho attempt with the given diameter and depth errors."""
    result = {
        "crater_diameter": TYCHO_DIAMETER * (1 + diameter_error_pct / 100.0),
        "crater_depth": TYCHO_DEPTH * (1 + depth_error_pct / 100.0),
        "energy_joules": 1e20,
    }
    return evaluate_challenge("tycho_crater", result, {"impact_angle": 45.0})


@pytest.mark.parametrize("depth_error_pct, diameter_error_pct, score, grade, passed", [
    (12.5,  0.0,  90.0, "S", True),    # exactly on the S line
    (12.51, 0.0,  90.0, "A", True),    # just below: 89.992, shown rounded as 90.0
    (31.25, 0.0,  75.0, "A", True),    # exactly on the A line
    (31.26, 0.0,  75.0, "B", True),    # just below: 74.992
    (12.5,  20.0, 60.0, "B", True),    # exactly on the B line (lowest passing score)
    (12.51, 20.0, 60.0, "C", False),   # just below: 59.992
])
def test_grade_boundaries(depth_error_pct, diameter_error_pct, score, grade, passed):
    evaluation = tycho(depth_error_pct, diameter_error_pct)
    assert evaluation["score"] == pytest.approx(score, abs=0.02)
    assert evaluation["grade"] == grade
    assert evaluation["passed"] is passed
    assert evaluation["xp_gain"] == GRADE_XP[grade]


def test_score_never_goes_below_zero():
    evaluation = tycho(1000.0, diameter_error_pct=1000.0)
    assert evaluation["score"] == 0.0
    assert evaluation["grade"] == "C"


# Regression test for a bug found by the boundary tests above: the depth score
# stops at 0, so an exact diameter alone used to score 60 and pass, however
# wrong the depth.  It was kept as a strict xfail until the fix landed.
def test_wildly_wrong_depth_does_not_pass():
    evaluation = tycho(900.0)   # depth ten times too deep
    assert evaluation["passed"] is False


@pytest.mark.parametrize("depth_error_pct, diameter_error_pct, grade, passed", [
    (49.99, 0.0, "B", True),     # depth score 0.02: still counts, 60.008 points
    (50.0,  0.0, "C", False),    # depth score 0: fails although the total is 60
    (0.0,  39.99, "C", False),   # diameter score 0.025: 40.015 points, fails on points
    (0.0,  40.0, "C", False),    # diameter score 0: fails outright
])
def test_a_measurement_scoring_zero_fails(depth_error_pct, diameter_error_pct, grade, passed):
    evaluation = tycho(depth_error_pct, diameter_error_pct)
    assert evaluation["grade"] == grade
    assert evaluation["passed"] is passed


@pytest.mark.parametrize("grade, previous_best, expected", [
    ("A", "B", 40),      # improving B -> A earns only the difference
    ("B", "A", 0),       # a worse grade earns nothing
    ("A", "A", 0),       # repeating the same grade earns nothing
    ("A", None, 100),    # first attempt earns the full amount
])
def test_repeat_attempts_only_earn_the_improvement(grade, previous_best, expected):
    assert award_challenge_xp(grade, previous_best) == expected


# --- Mission 3: energy threshold (1.0e23 J) and no depth target ----------

def dinosaur(energy: float) -> dict:
    result = {"crater_diameter": 150_000.0, "crater_depth": 0.0, "energy_joules": energy}
    return evaluate_challenge("dinosaur_extinction", result, {"impact_angle": 45.0})


def test_energy_exactly_at_threshold_has_no_penalty():
    evaluation = dinosaur(1.0e23)
    assert evaluation["score"] == 100.0
    assert evaluation["grade"] == "S"
    assert "Energy target reached" in evaluation["feedback"]


def test_energy_just_below_threshold_is_penalised():
    evaluation = dinosaur(0.99e23)
    # penalty = (1 - 0.99) * 40 = 0.4 points; the depth is not scored here
    assert evaluation["score"] == pytest.approx(99.6)
    assert evaluation["depth_error_pct"] == 0.0
    assert "below the extinction-level threshold" in evaluation["feedback"]


# --- Mission 2: maximum angle (20°) ---------------------------------------

def grazing(angle: float) -> dict:
    result = {"crater_diameter": 2_000.0, "crater_depth": 400.0, "energy_joules": 1e17}
    return evaluate_challenge("grazing_skimmer", result, {"impact_angle": angle})


def test_angle_exactly_at_limit_is_accepted():
    evaluation = grazing(20.0)
    assert evaluation["score"] == 100.0
    assert "A grazing" in evaluation["feedback"]


def test_angle_just_above_limit_is_penalised():
    evaluation = grazing(20.1)
    # penalty = 0.1° * 2.5 = 0.25 points
    assert evaluation["score"] == pytest.approx(99.8, abs=0.06)
    assert "above the mission's limit" in evaluation["feedback"]


def test_unknown_challenge_is_rejected():
    with pytest.raises(ValueError):
        evaluate_challenge("no_such_mission", {}, {})
