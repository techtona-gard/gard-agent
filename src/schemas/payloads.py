from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class OperationMode(str, Enum):
    INTERACTIVE = "INTERACTIVE"
    TELEMETRY_SYNC = "TELEMETRY_SYNC"

class IntentType(str, Enum):
    PROCESS_FOOD = "PROCESS_FOOD"
    PROCESS_LIFESTYLE = "PROCESS_LIFESTYLE"
    GENERAL_CHAT = "GENERAL_CHAT"
    UNKNOWN = "UNKNOWN"

class SleepData(BaseModel):
    duration_minutes: int
    quality_score: Optional[float] = None
    interruptions: Optional[int] = 0

class HealthData(BaseModel):
    hrv: Optional[float] = None
    avg_heart_rate: Optional[int] = None

class CalendarEvent(BaseModel):
    title: str
    start_time: str
    end_time: str
    is_stressful: Optional[bool] = False

class GerdQBaseline(BaseModel):
    score: int
    last_updated: str

class UserPayload(BaseModel):
    user_id: str
    message: Optional[str] = None
    image_url: Optional[str] = None
    sleep_data: Optional[SleepData] = None
    health_data: Optional[HealthData] = None
    calendar_events: Optional[List[CalendarEvent]] = []
    gerdq_baseline: Optional[GerdQBaseline] = None
    operation_mode: Optional[OperationMode] = None

class FoodAnalysis(BaseModel):
    food_name: str
    nutrition_facts: Dict[str, Any]
    gerd_trigger_level: str # HIGH, MEDIUM, LOW
    reasoning: str # Medically grounded via RAG

class LifestyleAnalysis(BaseModel):
    stress_level: str # HIGH, MEDIUM, LOW
    sleep_quality: str # POOR, FAIR, GOOD
    impact_on_gerd: str # RAG grounded justification

class ReminderItem(BaseModel):
    time: str = Field(description="Time format HH:MM")
    type: str = Field(description="Type of reminder, e.g. MEAL, SLEEP, MEDICATION")
    title: str = Field(description="Short title for notification")
    message: str = Field(description="Medically grounded reason/message for reminder")

class FinalResponse(BaseModel):
    risk_score: float
    sleep_quality: Optional[str] = None
    stress_level: Optional[str] = None
    recommendation: Optional[str] = None # Present only in INTERACTIVE mode
    user_message: Optional[str] = None    # Present only in INTERACTIVE mode
    reminders: Optional[List[ReminderItem]] = [] # Present only in INTERACTIVE mode
