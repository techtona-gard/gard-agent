from src.state import GraphState
from src.schemas import ScheduleResponse
from src.config import get_llm
from src.tools.rag_chroma import rag_tool

from datetime import datetime, timedelta, timezone

def run_schedule_node(state: GraphState) -> GraphState:
    sensor = state.get("sensor_data")
    date_str = state.get("date")
    
    if not date_str:
        # Jika dipanggil jam 23:59, defaultnya adalah generate untuk besok (WIB / UTC+7)
        tz_wib = timezone(timedelta(hours=7))
        tomorrow = datetime.now(tz_wib) + timedelta(days=1)
        date_str = tomorrow.strftime("%Y-%m-%d")
        
    baseline = state.get("baseline_gerd_q")
    
    baseline_status = "Low"
    if baseline:
        baseline_status = baseline.get("status", "Low") if isinstance(baseline, dict) else getattr(baseline, "status", "Low")
        
    events_str = "Tidak ada jadwal khusus."
    if sensor:
        event_list = sensor.get("event_schedule", []) if isinstance(sensor, dict) else getattr(sensor, "event_schedule", [])
        if event_list:
            events_str = ", ".join([e.get("title", "") if isinstance(e, dict) else getattr(e, "title", "") for e in event_list])
    
    rag_context = rag_tool.search("Jadwal makan ideal untuk penderita GERD")
    
    llm = get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(ScheduleResponse)
    
    prompt = f"""
    Kamu adalah GARD Schedule Agent untuk penderita GERD.
    Tugasmu adalah membuat jadwal makan yang aman berdasarkan jadwal aktivitas user.
    
    Tanggal Penjadwalan: {date_str}
    Aktivitas User: {events_str}
    Tingkat GERD (Baseline): {baseline_status}
    
    Panduan Medis (RAG): 
    {rag_context}
    
    Instruksi:
    Buatlah event kalender untuk Google Calendar (Sarapan, Snack, Makan Siang, Makan Malam dll).
    - start_time dan end_time harus berformat ISO 8601 UTC. Ingat, WIB adalah UTC+7. 
    - Misalnya untuk jam 08:00 WIB pagi tanggal {date_str}, format yang benar adalah '{date_str}T01:00:00Z' (dikurangi 7 jam).
    - Pastikan deskripsi padat berisi menu dan pantangan.
    """
    
    response = structured_llm.invoke(prompt)
    
    return {
        "schedule_response": response
    }
