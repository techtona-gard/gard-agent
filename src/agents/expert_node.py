from src.state import GraphState
from src.schemas import ChatResponse
from src.config import get_llm
from src.tools.rag_chroma import rag_tool

def calculate_risk_score(baseline: str, sleep_hours: float, hrv_status: str) -> int:
    score = 0
    if baseline.lower() == "high": score += 40
    else: score += 10
        
    if sleep_hours < 6.0: score += 30
    elif sleep_hours < 7.0: score += 15
        
    if hrv_status.lower() == "low": score += 30
    elif hrv_status.lower() == "high": score -= 10
        
    return min(max(score, 0), 100)

def run_expert_node(state: GraphState) -> GraphState:
    sensor = state.get("sensor_data")
    risk_score = calculate_risk_score(
        state.get("baseline_gerd_q") or "Low",
        sensor.sleep_hours if sensor else 8.0,
        sensor.hrv_status if sensor else "Normal"
    )
    
    rag_query = state.get("chat_input", "")
    rag_context = rag_tool.search(rag_query)
    
    llm = get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(ChatResponse)
    
    prompt = f"""
    Kamu adalah GARD-Expert Agent, asisten medis spesialis GERD.
    Balas input pengguna secara empatik dan medis.
    
    Konteks Pengguna:
    - Chat Summary Sebelumnya: {state.get('chat_summary', 'Belum ada.')}
    - Input Saat Ini: {state.get('chat_input', '')}
    - Risk Score: {risk_score} (Tinggi > 60, Sedang 30-60, Rendah < 30)
    
    Konteks Jurnal Medis (RAG):
    {rag_context}
    
    Tugas:
    1. Buat ai_message yang personal dan menjawab keluhan.
    2. Buat new_chat_summary 1-2 kalimat untuk merangkum chat lama dan pesan baru.
    """
    
    response = structured_llm.invoke(prompt)
    
    return {
        "chat_response": response,
        "chat_summary": response.new_chat_summary
    }
