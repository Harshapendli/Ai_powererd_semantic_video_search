"""
Quality Scoring Engine for Semanitc Footage Search.
Evaluates shots based on technical metrics (Stable, Sharp, Well-Lit) and assigns a 0-100 Quality Score.
Generates explainable feedback for editors.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def calculate_quality_score(metrics: Dict[str, Any]) -> Tuple[int, List[str]]:
    """
    Compute a deterministic quality score (0-100) and generate feedback tags.
    
    Expected metrics:
    - blur_var: float (Variance of Laplacian, higher = sharper)
    - avg_motion: float (Optical flow magnitude)
    - motion_variance: float (Stability metric, lower = smoother)
    - mean_brightness: float (0-255)
    - contrast: float (RMS contrast)
    """
    score = 100.0
    feedback = []

    # --- 1. Sharpness (Blur) ---
    # Typical Laplacian variance: < 100 is blurry, > 500 is sharp
    blur_var = metrics.get("blur_var", 500.0)
    if blur_var < 50.0:
        score -= 40
        feedback.append("⚠️ Extremely Blurry")
    elif blur_var < 150.0:
        score -= 20
        feedback.append("⚠️ Soft Focus")
    else:
        feedback.append("✅ Sharp")

    # --- 2. Stability (Motion) ---
    # High motion is fine (action), but irregular motion (shake) is bad.
    # We use motion_variance as a proxy for shake if available.
    motion_var = metrics.get("motion_variance", 0.0)
    avg_motion = metrics.get("avg_motion", 0.0)
    
    if motion_var > 10.0: # Arbitrary threshold, needs tuning
        score -= 20
        feedback.append("⚠️ Camera Shake")
    elif avg_motion > 5.0 and motion_var < 5.0:
        feedback.append("✅ Smooth Action")
    elif avg_motion < 0.5:
        feedback.append("ℹ️ Static Shot")

    # --- 3. Exposure / Lighting ---
    bright = metrics.get("mean_brightness", 127.0)
    if bright < 30:
        score -= 30
        feedback.append("⚠️ Underexposed")
    elif bright > 220:
        score -= 30
        feedback.append("⚠️ Overexposed")
    elif bright < 60:
        feedback.append("ℹ️ Low Key / Dark")
    elif bright > 180:
        feedback.append("ℹ️ High Key / Bright")
    else:
        feedback.append("✅ Well Exposed")

    # --- 4. Contrast ---
    contrast = metrics.get("contrast", 50.0)
    if contrast < 20.0:
        score -= 10
        feedback.append("⚠️ Low Contrast / Flat")

    # Clamp score
    final_score = int(max(0, min(100, score)))
    
    return final_score, feedback
