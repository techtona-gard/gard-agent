import logging

logger = logging.getLogger(__name__)

def calculate_gerd_risk_score(
    gerdq_score: int | None,
    food_trigger_level: str | None,
    stress_level: str | None,
    sleep_quality: str | None
) -> float:
    """
    Calculates overall GERD risk score (0 to 100) deterministically.
    - Clinical GERD-Q: Up to 50% risk. Score range is 4 to 18.
      Base Risk = ((gerdq_score - 4) / 14) * 50
    - Food: HIGH = +30%, MEDIUM = +15%, LOW = 0%
    - Stress: HIGH = +15%, MEDIUM = +7.5%, LOW = 0%
    - Sleep: POOR = +15%, FAIR = +7.5%, GOOD = 0%
    """
    # Defaults
    g_score = gerdq_score if gerdq_score is not None else 0
    # Standardize score to valid GERD-Q range [0, 18]
    g_score = max(0, min(18, g_score))
    
    base_risk = (g_score / 18.0) * 50.0
    
    food_mod = 0.0
    if food_trigger_level == "HIGH":
        food_mod = 30.0
    elif food_trigger_level == "MEDIUM":
        food_mod = 15.0
        
    stress_mod = 0.0
    if stress_level == "HIGH":
        stress_mod = 15.0
    elif stress_level == "MEDIUM":
        stress_mod = 7.5
        
    sleep_mod = 0.0
    if sleep_quality == "POOR":
        sleep_mod = 15.0
    elif sleep_quality == "FAIR":
        sleep_mod = 7.5
        
    total_risk = base_risk + food_mod + stress_mod + sleep_mod
    
    final_score = round(max(0.0, min(100.0, total_risk)), 2)
    logger.info(f"Rules: Risk score calculated deterministically. Base: {base_risk}%, Food Mod: {food_mod}%, Stress Mod: {stress_mod}%, Sleep Mod: {sleep_mod}% -> Final: {final_score}%")
    return final_score
