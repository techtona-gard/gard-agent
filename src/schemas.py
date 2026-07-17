from pydantic import BaseModel, Field
from typing import List, Optional

class SensorData(BaseModel):
    sleep_hours: float = Field(default=8.0, description="Jam tidur pengguna")
    hrv_status: str = Field(default="Normal", description="Status Heart Rate Variability (Normal, Low, High)")
    event_schedule: List[str] = Field(default_factory=list, description="Jadwal event atau makan hari ini")

class FoodProfile(BaseModel):
    food_name: str = Field(description="Nama makanan yang diekstrak")
    calories: float = Field(default=0.0)
    fat: float = Field(default=0.0)
    protein: float = Field(default=0.0)
    carbs: float = Field(default=0.0)
    is_gerd_trigger: bool = Field(default=False, description="Apakah makanan ini memicu GERD secara umum")

class MealSchedule(BaseModel):
    time: str = Field(description="Waktu makan (misal: 08:00)")
    meal_name: str = Field(description="Nama jadwal atau makanan")

class FinalResponse(BaseModel):
    ai_message: str = Field(description="Balasan chatbot yang personal untuk pengguna")
    meal_schedule: Optional[List[MealSchedule]] = Field(default=None, description="Waktu dan nama jadwal makan jika ada")
    new_chat_summary: str = Field(description="Ringkasan percakapan untuk memori loop berikutnya")

class AnalyzeRequest(BaseModel):
    thread_id: str
    user_id: str
    baseline_gerd_q: str = Field(default="Low")
    sensor_data: SensorData
    chat_input: str
