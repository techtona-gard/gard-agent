import json
import logging
from pydantic import BaseModel, Field
from typing import List, Optional
from src.state import GraphState
from src.schemas.payloads import FinalResponse, ReminderItem, OperationMode
from src.config import expert_llm, load_prompt, rag_service
from src.rules.gerd_rules import calculate_gerd_risk_score

logger = logging.getLogger(__name__)

# Structured representation of final response for Gemini
class ExpertResponse(BaseModel):
    recommendation: str = Field(description="Actionable diet/lifestyle recommendations grounded in medical evidence.")
    user_message: str = Field(description="Empathetic final response explaining findings in Indonesian.")
    reminders: List[ReminderItem] = Field(description="Specific timing reminders based on calendar schedule and medical guidelines.")

def generate_recommendation(state: GraphState) -> GraphState:
    """
    GERD Expert Agent:
    - Calculates Risk Score deterministically.
    - If OperationMode is TELEMETRY_SYNC: returns clean biometric results without LLM/RAG generation.
    - If OperationMode is INTERACTIVE: queries RAG medical guidelines, runs expert LLM reasoning,
      and generates final recommendations and student reminders.
    """
    logger.info("GERDExpert: Starting synthesizer...")
    payload = state["payload"]
    food_analysis = state.get("food_analysis")
    lifestyle_analysis = state.get("lifestyle_analysis")
    
    # 1. Deterministic risk calculation
    gerdq_score = payload.gerdq_baseline.score if payload.gerdq_baseline else None
    food_trigger = food_analysis.gerd_trigger_level if food_analysis else None
    stress = lifestyle_analysis.stress_level if lifestyle_analysis else None
    sleep = lifestyle_analysis.sleep_quality if lifestyle_analysis else None
    
    risk_score = calculate_gerd_risk_score(
        gerdq_score=gerdq_score,
        food_trigger_level=food_trigger,
        stress_level=stress,
        sleep_quality=sleep
    )
    
    # 2. Check Operation Mode
    mode = payload.operation_mode or OperationMode.INTERACTIVE
    
    if mode == OperationMode.TELEMETRY_SYNC:
        logger.info("GERDExpert: TELEMETRY_SYNC mode resolved. Bypassing LLM generation.")
        # Purely return calculated biometric metrics without chatbot message/reminders
        response_data = {
            "risk_score": risk_score,
            "sleep_quality": sleep,
            "stress_level": stress,
            "recommendation": None,
            "user_message": None,
            "reminders": []
        }
        return {"final_response": response_data}

    # === INTERACTIVE MODE WORKFLOW (Requires LLM and RAG) ===
    logger.info("GERDExpert: INTERACTIVE mode resolved. Executing RAG reasoning...")
    
    # Compile queries for RAG context
    query_terms = []
    if food_analysis:
        query_terms.append(food_analysis.food_name)
        if food_analysis.gerd_trigger_level == "HIGH":
            query_terms.append("lemak asam pedas LES lambung")
    if lifestyle_analysis:
        if lifestyle_analysis.stress_level == "HIGH":
            query_terms.append("stres cemas LES sensitivitas lambung")
        if lifestyle_analysis.sleep_quality == "POOR":
            query_terms.append("begadang kurang tidur irama sirkadian gerd")
            
    rag_query = " ".join(query_terms) if query_terms else "saran mencegah gerd mahasiswa"
    medical_context = rag_service.query_context(rag_query, n_results=2)
    
    # LLM Call for Synthesizing Recommendation & Calendar Reminders
    prompt_template = load_prompt("gerd_expert_prompt.txt")
    
    # Construct complete structured analysis payload for LLM to reason upon
    calendar_info = [{"title": e.title, "start_time": e.start_time, "end_time": e.end_time} for e in payload.calendar_events] if payload.calendar_events else []
    
    llm_context = {
        "user_message": payload.message,
        "gerdq_baseline_score": gerdq_score,
        "calculated_risk_score": risk_score,
        "food_analysis": food_analysis.dict() if food_analysis else "None",
        "lifestyle_analysis": lifestyle_analysis.dict() if lifestyle_analysis else "None",
        "student_calendar_events": calendar_info,
        "medical_context_rag": medical_context
    }
    
    system_input = (
        f"{prompt_template}\n\n"
        f"Input Data:\n{json.dumps(llm_context, indent=2)}\n\n"
        f"Generate the response conforming to ExpertResponse schema."
    )
    
    try:
        structured_llm = expert_llm.with_structured_output(ExpertResponse)
        prediction = structured_llm.invoke(system_input)
        
        # Format reminders
        reminders_list = []
        for rem in prediction.reminders:
            reminders_list.append(ReminderItem(
                time=rem.time,
                type=rem.type,
                title=rem.title,
                message=rem.message
            ))
            
        response_data = {
            "risk_score": risk_score,
            "sleep_quality": sleep,
            "stress_level": stress,
            "recommendation": prediction.recommendation,
            "user_message": prediction.user_message,
            "reminders": [r.dict() for r in reminders_list]
        }
    except Exception as e:
        logger.error(f"GERDExpert: LLM recommendation synthesis failed: {str(e)}. Using fallback.")
        
        # Empathetic local fallback
        fallback_msg = f"Halo! Berdasarkan analisis kami, tingkat risiko GERD Anda saat ini adalah {risk_score}%. Silakan ikuti rekomendasi berikut."
        fallback_rec = "Konsumsi makanan porsi kecil tapi sering, kelola tingkat stres Anda, dan tidur dengan bantal ditinggikan jika bergejala."
        
        # Basic dynamic reminder fallback if calendar has stressful items
        fallback_reminders = []
        if payload.calendar_events:
            for event in payload.calendar_events:
                if event.is_stressful:
                    fallback_reminders.append(ReminderItem(
                        time=event.start_time,
                        type="RELAKSASI",
                        title=f"Rileks sebelum {event.title}",
                        message="Lakukan teknik napas dalam (deep breathing) 5 menit sebelum aktivitas dimulai untuk menurunkan stres pemicu LES."
                    ))
                    
        response_data = {
            "risk_score": risk_score,
            "sleep_quality": sleep,
            "stress_level": stress,
            "recommendation": fallback_rec,
            "user_message": fallback_msg,
            "reminders": [r.dict() for r in fallback_reminders]
        }
        
    logger.info("GERDExpert: Synthesis workflow completed successfully.")
    return {"final_response": response_data}