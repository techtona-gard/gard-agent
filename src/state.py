from typing import TypedDict, Optional
from src.schemas import SensorData, FoodProfile, ChatResponse, ScanFoodResponse, ScheduleResponse

class GraphState(TypedDict):
    action: str  # "chat", "scan", "schedule"
    thread_id: str
    user_id: str
    
    # Input data
    chat_input: Optional[str]
    image_base64: Optional[str]
    baseline_gerd_q: Optional[str]
    sensor_data: Optional[SensorData]
    date: Optional[str]
    
    # Processed Data
    food_analysis: Optional[FoodProfile]
    
    # Memory
    chat_summary: str
    
    # Final Outputs
    chat_response: Optional[ChatResponse]
    scan_response: Optional[ScanFoodResponse]
    schedule_response: Optional[ScheduleResponse]
