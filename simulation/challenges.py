"""
Scientific Gamification & Lunar Impact Missions Module.

Provides game-like educational challenge missions where users adjust real
physical impact parameters to match known astronomical landmarks or planetary
defense objectives, with instant scientific diagnostics and scoring.
"""

from typing import Optional


CHALLENGES = [
    {
        "id": "tycho_crater",
        "title": "Mission 1: Replicate Tycho Crater",
        "difficulty": "Medium ⭐⭐",
        "badge": "🌕 High Highland Master",
        "description": "Tycho is one of the best-known young complex craters in the Moon's southern highlands (about 108 million years old): about 85 km across and 4.8 km deep, with a prominent central peak. Adjust the asteroid's diameter and velocity to reproduce its size!",
        "target_diameter": 85000.0,  # 85 km
        "target_depth": 4800.0,      # 4.8 km
        "target_angle": 45.0,
        "diameter_tolerance_pct": 10.0,
        "depth_tolerance_pct": 15.0,
        "tips": "Tip: on the Moon, craters wider than about 15 km collapse into a flat floor with a central peak (complex craters). An asteroid about 5–7 km across at 15–18 km/s reaches Tycho's size.",
    },
    {
        "id": "grazing_skimmer",
        "title": "Mission 2: Low-Angle Grazing Impact",
        "difficulty": "Hard ⭐⭐⭐",
        "badge": "☄️ Grazing Deflector",
        "description": "Try a very shallow impact (angle ≤ 20°)! Below 25° a \"forbidden zone\" with no ejecta appears uprange; below 15° the crater starts to stretch into an ellipse along the flight path. Make an oblique crater about 2 km across.",
        "target_diameter": 2000.0,   # 2 km
        "target_depth": 400.0,       # fresh simple craters: d ≈ 0.2 D (Pike 1977)
        "max_angle": 20.0,
        "diameter_tolerance_pct": 20.0,
        "depth_tolerance_pct": 25.0,
        "tips": "Tip: only the vertical component of an oblique impact excavates, so the crater is smaller than a vertical impact's. Try a 60–100 m asteroid at 18 km/s and 10°–15°; below 10° the ejecta forms a \"butterfly\" pattern.",
    },
    {
        "id": "dinosaur_extinction",
        "title": "Mission 3: Chicxulub Cataclysm Energy",
        "difficulty": "Extreme ⭐⭐⭐⭐",
        "badge": "🦖 Supernova Impact",
        "description": "Recreate the energy of the Chicxulub impact that struck Earth about 66 million years ago (kinetic energy ≥ 1.0 × 10²³ J, crater ≥ 150 km across)!",
        "min_energy": 1.0e23,
        "target_diameter": 150000.0, # 150 km
        "diameter_tolerance_pct": 25.0,
        "tips": "Tip: it takes a giant asteroid around 10 km across at about 20 km/s. On the Moon this makes a complex crater more than 150 km across.",
    },
    {
        "id": "apollo_tranquillity",
        "title": "Mission 4: Mare Tranquillitatis Micro-Crater",
        "difficulty": "Easy ⭐",
        "badge": "🚀 Apollo Scout",
        "description": "The plain of Mare Tranquillitatis, where Apollo 11 landed, is dotted with small bowl-shaped craters a few hundred metres across. Use a meteoroid just tens of metres across to make a bowl-shaped crater about 300 m across and 60 m deep!",
        "target_diameter": 300.0,
        "target_depth": 60.0,
        "diameter_tolerance_pct": 12.0,
        "depth_tolerance_pct": 18.0,
        "tips": "Tip: a crater is roughly 60 times the size of its impactor. Try a 4–6 m meteoroid at an ordinary 15–20 km/s (the Moon's mean impact speed is about 18 km/s).",
    },
]


# XP for reaching each grade.  Repeat attempts only earn the difference when
# they beat the player's previous best grade (see award_challenge_xp).
GRADE_XP = {"S": 150, "A": 100, "B": 60, "C": 25}


def award_challenge_xp(grade: str, previous_best: Optional[str]) -> int:
    """XP for this attempt: the gain over the best grade already earned."""
    already = GRADE_XP.get(previous_best, 0) if previous_best else 0
    return max(0, GRADE_XP[grade] - already)


def get_challenge_by_id(challenge_id: str) -> Optional[dict]:
    """Lookup challenge config by ID."""
    for ch in CHALLENGES:
        if ch["id"] == challenge_id:
            return ch
    return None


def evaluate_challenge(challenge_id: str, result: dict, params: dict) -> dict:
    """
    Evaluate user's simulation against target challenge parameters.

    Returns:
        score: float (0 - 100)
        grade: str ('S', 'A', 'B', 'C')
        diameter_error_pct: float
        depth_error_pct: float
        xp_gain: int
        feedback: str
        passed: bool
    """
    ch = get_challenge_by_id(challenge_id)
    if not ch:
        raise ValueError(f"Unknown challenge: {challenge_id}")

    crater_diam = result.get("crater_diameter", 0.0)
    crater_depth = result.get("crater_depth", 0.0)
    energy = result.get("energy_joules", 0.0)
    angle = params.get("impact_angle", 45.0)

    # 1. Diameter error
    target_diam = ch.get("target_diameter")
    if target_diam:
        diam_err_pct = abs(crater_diam - target_diam) / target_diam * 100.0
    else:
        diam_err_pct = 0.0

    # 2. Depth error
    target_depth = ch.get("target_depth")
    if target_depth:
        depth_err_pct = abs(crater_depth - target_depth) / target_depth * 100.0
    else:
        depth_err_pct = 0.0

    # 3. Special condition checks
    penalty = 0.0
    feedback_notes = []

    if "max_angle" in ch and angle > ch["max_angle"]:
        penalty += (angle - ch["max_angle"]) * 2.5
        feedback_notes.append(f"Impact angle {angle}° is above the mission's limit of {ch['max_angle']}°.")
    elif "max_angle" in ch:
        feedback_notes.append(f"A grazing {angle}° impact: extreme asymmetric terrain achieved!")

    if "min_energy" in ch:
        if energy < ch["min_energy"]:
            energy_ratio = energy / ch["min_energy"]
            penalty += (1.0 - energy_ratio) * 40.0
            feedback_notes.append("The impact energy is below the extinction-level threshold (10²³ J): increase the asteroid's diameter or velocity.")
        else:
            feedback_notes.append("Energy target reached: the impact breaks the 10²³ J extinction-level threshold!")

    # Base score from diameter and depth accuracy
    diam_score = max(0.0, 100.0 - diam_err_pct * 2.5)
    depth_score = max(0.0, 100.0 - depth_err_pct * 2.0)
    
    if target_depth:
        raw_score = 0.6 * diam_score + 0.4 * depth_score - penalty
    else:
        raw_score = diam_score - penalty

    score = max(0.0, min(100.0, raw_score))

    # Grade determination
    if score >= 90.0:
        grade = "S"
        passed = True
        status_msg = "Outstanding! Every quantity matches the known lunar feature."
    elif score >= 75.0:
        grade = "A"
        passed = True
        status_msg = "Mission accomplished! The values are very close to the target."
    elif score >= 60.0:
        grade = "B"
        passed = True
        status_msg = "Passed. Fine-tune the parameters for a closer match."
    else:
        grade = "C"
        passed = False
        status_msg = "Not passed. The values are far off: see the tip and adjust the asteroid's mass and velocity."

    # A measurement that scores zero fails the mission outright.  Without this,
    # an exact diameter alone reaches 0.6 * 100 = 60 points and passes, however
    # wrong the depth is, because the depth score cannot drop below zero.
    if diam_score == 0.0 or (target_depth and depth_score == 0.0):
        grade = "C"
        passed = False
        status_msg = "Not passed. A diameter error of 40% or more, or a depth error of 50% or more, fails the mission outright."

    # One sentence per part, so the interface can translate each on its own
    feedback_parts = [status_msg, f"Diameter error: {diam_err_pct:.1f}%, depth error: {depth_err_pct:.1f}%.", *feedback_notes]

    return {
        "challenge_id": challenge_id,
        "challenge_title": ch["title"],
        "score": round(score, 1),
        "grade": grade,
        "passed": passed,
        "diameter_error_pct": round(diam_err_pct, 2),
        "depth_error_pct": round(depth_err_pct, 2),
        "xp_gain": GRADE_XP[grade],   # before repeat-attempt rules; the API adjusts it
        "feedback": " ".join(feedback_parts),
        "feedback_parts": feedback_parts,
        "badge": ch.get("badge"),
    }
