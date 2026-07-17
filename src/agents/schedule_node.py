from src.state import GraphState
from src.schemas import ScheduleResponse
from src.config import get_llm
from src.tools.rag_chroma import rag_tool

def run_schedule_node(state: GraphState) -> GraphState:
    sensor = state.get("sensor_data")
    date = state.get("date", "2023-10-25")
    events = ", ".join(sensor.event_schedule) if sensor and sensor.event_schedule else "Tidak ada jadwal khusus."
    
    rag_context = rag_tool.search("Jadwal makan ideal untuk penderita GERD")
    
    llm = get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(ScheduleResponse)
    
    prompt = f"""
    Kamu adalah GARD Schedule Agent untuk penderita GERD.
    Tugasmu adalah membuat jadwal makan yang aman berdasarkan jadwal aktivitas user.
    
    Tanggal: {date}
    Aktivitas User Hari Ini: {events}
    Tingkat GERD (Baseline): {state.get('baseline_gerd_q', 'Low')}
    
    Panduan Medis (RAG): 
    {rag_context}
    
    Instruksi:
    Buatlah event kalender untuk Google Calendar (Sarapan, Snack, Makan Siang, Makan Malam dll).
    - start_time dan end_time harus berformat ISO 8601 UTC. Misalnya untuk jam 08:00 WIB tanggal {date}, formatnya '{date}T01:00:00Z'.
    - Pastikan deskripsi padat berisi menu dan pantangan.
    """
    
    response = structured_llm.invoke(prompt)
    
    return {
        "schedule_response": response
    }
