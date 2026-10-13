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
        "title": "Mission 1: Replicate Tycho Crater (精準重現第谷坑)",
        "difficulty": "Medium ⭐⭐",
        "badge": "🌕 High Highland Master",
        "description": "第谷坑 (Tycho Crater) 是月球南部高地最著名的年輕撞擊坑之一，直徑約 85 km，深達 4.8 km。請調整小行星直徑與撞擊速度，嘗試精確重現它的規模！",
        "target_diameter": 85000.0,  # 85 km
        "target_depth": 4800.0,      # 4.8 km
        "target_angle": 45.0,
        "diameter_tolerance_pct": 10.0,
        "depth_tolerance_pct": 15.0,
        "tips": "提示：在月球重力 (1.62 m/s²) 下，直徑 2000m~3000m 的小行星以約 15~18 km/s 撞擊能達到類似規模。",
    },
    {
        "id": "grazing_skimmer",
        "title": "Mission 2: Low-Angle Grazing Impact (低空掠射橢圓坑)",
        "difficulty": "Hard ⭐⭐⭐",
        "badge": "☄️ Grazing Deflector",
        "description": "挑戰超淺角撞擊 (撞擊角 ≤ 20°)，利用斜向撞擊物理模型製造出具有高度方向性偏心率的非對稱橢圓坑與順風向噴射毯！",
        "target_diameter": 2000.0,   # 2 km
        "target_depth": 350.0,
        "max_angle": 20.0,
        "diameter_tolerance_pct": 20.0,
        "depth_tolerance_pct": 25.0,
        "tips": "提示：角度越小（例如 10°~15°），非對稱橢圓變形與順風拋射物越顯著。",
    },
    {
        "id": "dinosaur_extinction",
        "title": "Mission 3: Chicxulub Cataclysm Energy (恐龍滅絕級撞擊能量)",
        "difficulty": "Extreme ⭐⭐⭐⭐",
        "badge": "🦖 Supernova Impact",
        "description": "重現約 6600 萬年前希克蘇魯伯小行星撞擊地球時的滅世能量釋放（撞擊動能 ≥ 1.0 × 10²³ 焦耳，撞擊坑直徑 ≥ 150 km）！",
        "min_energy": 1.0e23,
        "target_diameter": 150000.0, # 150 km
        "diameter_tolerance_pct": 25.0,
        "tips": "提示：需要 10 km 級別的巨型小行星並以超過 20 km/s 的超高速撞擊。",
    },
    {
        "id": "apollo_tranquillity",
        "title": "Mission 4: Mare Tranquillitatis Micro-Crater (靜海微型坑精確落點)",
        "difficulty": "Easy ⭐",
        "badge": "🚀 Apollo Scout",
        "description": "阿波羅 11 號著陸的靜海平原散佈著直徑數百公尺的小型碗狀坑。請調配一顆直徑僅數十公尺的小隕石，精準製造一個直徑約 300 公尺、深度約 60 公尺的小型碗狀坑！",
        "target_diameter": 300.0,
        "target_depth": 60.0,
        "diameter_tolerance_pct": 12.0,
        "depth_tolerance_pct": 18.0,
        "tips": "提示：嘗試 5m~12m 的小隕石與 15~20 km/s 的常規衝擊速度。",
    },
]


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
        feedback_notes.append(f"撞擊角度為 {angle}°，高於任務要求的上限 {ch['max_angle']}°。")
    elif "max_angle" in ch:
        feedback_notes.append(f"成功採用 {angle}° 低掠射角，達成極端非對稱地形！")

    if "min_energy" in ch:
        if energy < ch["min_energy"]:
            energy_ratio = energy / ch["min_energy"]
            penalty += (1.0 - energy_ratio) * 40.0
            feedback_notes.append("撞擊動能尚未達到滅絕級能量閾值 (10²³ J)，請提升小行星直徑或速度。")
        else:
            feedback_notes.append("巨大能量釋放達標！突破 10²³ 焦耳滅世級衝擊門檻。")

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
        xp_gain = 150
        passed = True
        status_msg = "卓越達成！各項物理量完美吻合已知月球地質特徵！"
    elif score >= 75.0:
        grade = "A"
        xp_gain = 100
        passed = True
        status_msg = "任務成功！物理數值極其接近目標範圍。"
    elif score >= 60.0:
        grade = "B"
        xp_gain = 60
        passed = True
        status_msg = "基本達標。可微調參數以獲得更精確的模擬匹配。"
    else:
        grade = "C"
        xp_gain = 25
        passed = False
        status_msg = "未達標準。數值偏差較大，請參考科學提示重新調整小行星質量與速度。"

    feedback = f"{status_msg} 坑徑誤差: {diam_err_pct:.1f}%，深度誤差: {depth_err_pct:.1f}%。"
    if feedback_notes:
        feedback += " " + " ".join(feedback_notes)

    return {
        "challenge_id": challenge_id,
        "challenge_title": ch["title"],
        "score": round(score, 1),
        "grade": grade,
        "passed": passed,
        "diameter_error_pct": round(diam_err_pct, 2),
        "depth_error_pct": round(depth_err_pct, 2),
        "xp_gain": xp_gain,
        "feedback": feedback,
        "badge": ch.get("badge"),
    }
