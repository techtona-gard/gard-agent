from src.state import GraphState
from src.schemas import FinalResponse
from src.config import get_llm
from src.tools.rag_chroma import rag_tool

def calculate_risk_score(baseline: str, sleep_hours: float, hrv_status: str) -> int:
    score = 0
    # Baseline
    if baseline.lower() == "high":
        score += 40
    else:
        score += 10
        
    # Sleep
    if sleep_hours < 6.0:
        score += 30
    elif sleep_hours < 7.0:
        score += 15
        
    # HRV
    if hrv_status.lower() == "low":
        score += 30
    elif hrv_status.lower() == "high":
        score -= 10
        
    return min(max(score, 0), 100)

def run_expert_node(state: GraphState) -> GraphState:
    # A. Risk Scoring (Non-LLM)
    risk_score = calculate_risk_score(
        state["baseline_gerd_q"],
        state["sensor_data"].sleep_hours,
        state["sensor_data"].hrv_status
    )
    
    # B. RAG Retrieval
    rag_query = state["chat_input"]
    if state.get("food_analysis"):
        rag_query += f" efek konsumsi {state['food_analysis'].food_name} bagi asam lambung dan GERD"
        
    rag_context = rag_tool.search(rag_query)
    
    # C. LLM Reasoning & Output
    llm = get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(FinalResponse)
    
    food_info = ""
    if state.get("food_analysis"):
        f = state["food_analysis"]
        food_info = f"Data Makanan Terdeteksi: {f.food_name}, Kalori: {f.calories} kcal, Memicu GERD: {f.is_gerd_trigger}"
        
    prompt = f"""
    Kamu adalah GARD-Expert Agent, asisten medis spesialis GERD.
    Tugasmu adalah menganalisis kondisi pengguna dan memberikan balasan yang empatik serta medis.
    
    Konteks Pengguna:
    - Chat Summary Sebelumnya: {state.get('chat_summary', 'Belum ada percakapan.')}
    - Input Saat Ini: {state['chat_input']}
    - Risk Score Saat Ini (0-100): {risk_score} (Interpretasi: Tinggi > 60, Sedang 30-60, Rendah < 30)
    - Data Sensor Terkini: Tidur {state['sensor_data'].sleep_hours} jam, HRV: {state['sensor_data'].hrv_status}
    - Jadwal/Event Hari Ini: {', '.join(state['sensor_data'].event_schedule) if state['sensor_data'].event_schedule else 'Tidak ada'}
    
    {food_info}
    
    Konteks Referensi Jurnal Medis (RAG):
    {rag_context}
    
    Instruksi Pembuatan Output:
    1. ai_message: Balas input pengguna secara langsung, personal, dan empatik. Beri penjelasan medis singkat berdasarkan RAG atau Risk Score jika relevan.
    2. meal_schedule: Buat jadwal makan (waktu & nama jadwal) jika pengguna menanyakan jadwal atau jika kamu merasa perlu merekomendasikan jam makan ideal bagi penderita GERD hari ini.
    3. new_chat_summary: Buat 1-2 kalimat ringkasan yang merangkum chat summary lama DAN interaksi pada giliran ini. Ini penting agar agen tidak lupa konteks di pesan berikutnya.
    """
    
    response = structured_llm.invoke(prompt)
    
    # D. Memory Update
    return {
        "final_response": response,
        "chat_summary": response.new_chat_summary
    }
