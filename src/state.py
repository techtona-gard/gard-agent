from typing import TypedDict, Dict, Any, Optional


# ==========================================================
# Shared Data Structures
# ==========================================================

class Nutrition(TypedDict):
    """
    Ringkasan nutrisi makanan.
    Nilai menggunakan estimasi per porsi.
    """
    calories: float
    protein: float
    fat: float
    carbohydrate: float


class FoodAnalysis(TypedDict):
    """
    Hasil analisis Nutri Scan Agent.
    """

    detected_food: str

    nutrition: Nutrition

    gerd_triggers: list[str]

    confidence: float

    risk_level: str


class LifestyleAnalysis(TypedDict):
    """
    Hasil analisis Bio Rhythm Agent.
    """

    sleep_quality: str

    stress_level: str

    activity_level: str

    notes: str


# ==========================================================
# Global LangGraph State
# ==========================================================

class GardAgentState(TypedDict):

    # ======================================================
    # Input dari Platform
    # ======================================================

    user_message: str

    image_path: Optional[str]

    baseline_gerdq: str

    vitals_data: Dict[str, Any]

    activity_data: Dict[str, Any]

    # ======================================================
    # Antar Agent
    # ======================================================

    input_route: str

    food_analysis: FoodAnalysis

    lifestyle_analysis: LifestyleAnalysis

    # ======================================================
    # Output
    # ======================================================

    risk_score: int

    medical_context: str

    final_response: str