from pydantic import BaseModel, Field
from typing import List, Optional

# --- Shared Models ---
class SensorData(BaseModel):
    sleep_hours: float = Field(default=8.0, description="Jam tidur pengguna")
    hrv_status: str = Field(default="Normal", description="Status Heart Rate Variability (Normal, Low, High)")
    event_schedule: List[str] = Field(default_factory=list, description="Jadwal pengguna hari ini")

class FoodProfile(BaseModel):
    food_name: str = Field(description="Nama makanan yang diekstrak")
    calories: float = Field(default=0.0)
    fat: float = Field(default=0.0)
    protein: float = Field(default=0.0)
    carbs: float = Field(default=0.0)
    is_gerd_trigger: bool = Field(default=False, description="Apakah memicu GERD")
    reasoning: str = Field(default="", description="Alasan medis kenapa memicu atau aman")

# --- CHAT FEATURE ---
class ChatRequest(BaseModel):
    thread_id: str
    user_id: str
    baseline_gerd_q: str = Field(default="Low")
    sensor_data: SensorData
    chat_input: str

class ChatResponse(BaseModel):
    ai_message: str = Field(description="Balasan chatbot yang personal untuk pengguna")
    new_chat_summary: str = Field(description="Ringkasan percakapan untuk memori loop berikutnya")

# --- SCAN FOOD FEATURE ---
class ScanFoodRequest(BaseModel):
    thread_id: str
    user_id: str
    image_base64: Optional[str] = Field(default=None, description="Gambar makanan dalam base64 (tanpa prefix header)")
    chat_input: Optional[str] = Field(default=None, description="Pertanyaan opsional atau deskripsi makanan")

class ScanFoodResponse(BaseModel):
    ai_message: str = Field(description="Penjelasan AI mengenai makanan tersebut")
    food_profile: Optional[FoodProfile] = None

# --- SCHEDULE FEATURE ---
class ScheduleRequest(BaseModel):
    thread_id: str
    user_id: str
    baseline_gerd_q: str = Field(default="Low")
    sensor_data: SensorData
    date: str = Field(default="2023-10-25", description="Tanggal untuk jadwal YYYY-MM-DD")

class ScheduleEvent(BaseModel):
    summary: str = Field(description="Judul event kalender (misal: 'Sarapan Aman Lambung')")
    description: str = Field(description="Deskripsi detail, menu rekomendasi, pantangan, alasan")
    start_time: str = Field(description="Format ISO 8601 UTC, misal: '2023-10-25T01:00:00Z'")
    end_time: str = Field(description="Format ISO 8601 UTC, misal: '2023-10-25T01:30:00Z'")

class ScheduleResponse(BaseModel):
    events: List[ScheduleEvent] = Field(description="Daftar jadwal untuk diinput ke Google Calendar")
    ai_message: str = Field(description="Pesan pengantar dari AI mengenai jadwal yang dibuat")
