from typing import TypedDict, Optional
from src.schemas import SensorData, FoodProfile, FinalResponse

class GraphState(TypedDict):
    thread_id: str
    user_id: str
    baseline_gerd_q: str  # High/Low
    sensor_data: SensorData
    chat_input: str
    food_analysis: Optional[FoodProfile]
    chat_summary: str
    final_response: Optional[FinalResponse]
