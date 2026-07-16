import logging
from pydantic import BaseModel, Field
from src.state import GraphState
from src.schemas.payloads import LifestyleAnalysis
from src.config import nutrition_llm, load_prompt, rag_service

logger = logging.getLogger(__name__)

# Output schema for BioRhythm classification
class BioRhythmClassification(BaseModel):
    sleep_quality: str = Field(description="Sleep rating: GOOD, FAIR, POOR")
    stress_level: str = Field(description="Stress rating: LOW, MEDIUM, HIGH")
    impact_on_gerd: str = Field(description="Medically grounded reason in Indonesian citing RAG details.")

def process_lifestyle(state: GraphState) -> GraphState:
    """
    Bio Rhythm Agent:
    - Parses sleep duration/interruptions, HRV, and calendar events.
    - Queries ChromaDB RAG for biometrics medical references.
    - Employs Gemini to classify sleep_quality and stress_level based on RAG literature context.
    """
    logger.info("BioRhythm: Processing lifestyle telemetry data...")
    payload = state["payload"]
    
    # 1. Parse biometrics from payload
    sleep_duration = payload.sleep_data.duration_minutes if payload.sleep_data else 480
    sleep_interruptions = payload.sleep_data.interruptions if payload.sleep_data else 0
    hrv = payload.health_data.hrv if payload.health_data else 65.0
    avg_hr = payload.health_data.avg_heart_rate if payload.health_data else 70
    
    calendar_stress_events = 0
    if payload.calendar_events:
        calendar_stress_events = len([e for e in payload.calendar_events if e.is_stressful])
        
    # 2. Query RAG for medical biometrics guidelines
    rag_query = "durasi tidur interupsi hrv detak jantung stres lambung gerd acg"
    rag_context = rag_service.query_context(rag_query, n_results=2)
    logger.info("BioRhythm: Retrived biometrics guidelines context from RAG database.")
    
    # 3. LLM Classification based on RAG context (Medically Grounded - No Guessing)
    try:
        prompt_template = load_prompt("bio_rhythm_prompt.txt")
        formatted_prompt = prompt_template.format(
            sleep_duration=sleep_duration,
            sleep_interruptions=sleep_interruptions,
            hrv=hrv,
            avg_hr=avg_hr,
            calendar_stress_events=calendar_stress_events,
            rag_context=rag_context
        )
        
        structured_llm = nutrition_llm.with_structured_output(BioRhythmClassification)
        classification = structured_llm.invoke(formatted_prompt)
        
        sleep_quality = classification.sleep_quality
        stress_level = classification.stress_level
        impact_on_gerd = classification.impact_on_gerd
    except Exception as e:
        logger.error(f"BioRhythm: LLM biometrics classification failed: {str(e)}. Using fallback rules.")
        # Strict fallback matching ACG standards if LLM fails
        sleep_quality = "POOR" if (sleep_duration < 360 or sleep_interruptions > 3) else ("FAIR" if sleep_duration < 420 else "GOOD")
        stress_level = "HIGH" if (hrv < 40 or avg_hr > 85 or calendar_stress_events >= 2) else "LOW"
        impact_on_gerd = "Pedoman ACG: Kurang tidur dan stres psikologis melemahkan LES dan meningkatkan sensitivitas asam lambung."
        
    analysis = LifestyleAnalysis(
        stress_level=stress_level,
        sleep_quality=sleep_quality,
        impact_on_gerd=impact_on_gerd
    )
    
    logger.info(f"BioRhythm: Classification complete. Sleep: {sleep_quality}, Stress: {stress_level}")
    return {"lifestyle_analysis": analysis}