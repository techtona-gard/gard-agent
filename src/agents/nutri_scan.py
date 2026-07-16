import logging
from pydantic import BaseModel, Field
from src.state import GraphState
from src.schemas.payloads import FoodAnalysis
from src.config import nutrition_llm, load_prompt, fatsecret_service, rag_service
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# Output schema for initial extraction (ONLY food name)
class FoodNameExtraction(BaseModel):
    food_name: str = Field(description="The primary food/drink item name extracted.")

# Output schema for final trigger classification
class FoodTriggerClassification(BaseModel):
    gerd_trigger_level: str = Field(description="Trigger level: HIGH, MEDIUM, LOW")
    reasoning: str = Field(description="Medically grounded reason in Indonesian citing RAG details.")

def process_food(state: GraphState) -> GraphState:
    """
    Nutri Scan Agent:
    - Phase 1: Gemini reads inputs and extracts ONLY the name of the food.
    - Phase 2: Queries FatSecret API for food nutrition parameters.
    - Phase 3: Queries ChromaDB RAG for medical guidelines related to these nutrients.
    - Phase 4: Gemini classifies the trigger level based on RAG guidelines.
    """
    logger.info("NutriScan: Starting food analysis workflow...")
    payload = state["payload"]
    
    # === PHASE 1: Extract ONLY Food Name ===
    food_name = "Unknown Food"
    try:
        prompt_extract = load_prompt("nutri_scan_prompt.txt")
        structured_extract = nutrition_llm.with_structured_output(FoodNameExtraction)
        
        if payload.image_url:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_extract},
                    {"type": "image_url", "image_url": {"url": payload.image_url}}
                ]
            )
            extracted = structured_extract.invoke([message])
        else:
            extracted = structured_extract.invoke(f"{prompt_extract}\n\nUser input: {payload.message}")
            
        food_name = extracted.food_name
    except Exception as e:
        logger.error(f"NutriScan: Extraction failed: {str(e)}. Attempting heuristic fallback.")
        if payload.message:
            msg_lower = payload.message.lower()
            for candidate in ["indomie", "kopi", "teh", "ayam", "jus jeruk", "cokelat", "seblak"]:
                if candidate in msg_lower:
                    food_name = candidate
                    break

    logger.info(f"NutriScan: Food name resolved to: '{food_name}'")

    # === PHASE 2: FatSecret Query ===
    nutrition = fatsecret_service.search_food_nutrition(food_name)
    logger.info(f"NutriScan: FatSecret nutrients: {nutrition}")

    # === PHASE 3: Query RAG for Medical Guidelines ===
    rag_query = f"{food_name} lemak pedas asam kafein lambung pemicu gerd"
    rag_context = rag_service.query_context(rag_query, n_results=2)
    logger.info("NutriScan: Retrieved medical guidelines context from ChromaDB.")

    # === PHASE 4: Medically Grounded Trigger Classification ===
    try:
        prompt_classify = load_prompt("nutri_classify_prompt.txt")
        formatted_classify_prompt = prompt_classify.format(
            food_name=food_name,
            calories=nutrition.get("calories", 0.0),
            fat=nutrition.get("fat", 0.0),
            spicy=nutrition.get("spicy", False),
            acidic=nutrition.get("acidic", False),
            caffeine=nutrition.get("caffeine", False),
            chocolate=nutrition.get("chocolate", False),
            rag_context=rag_context
        )
        
        structured_classify = nutrition_llm.with_structured_output(FoodTriggerClassification)
        classification = structured_classify.invoke(formatted_classify_prompt)
        
        trigger_level = classification.gerd_trigger_level
        reasoning = classification.reasoning
    except Exception as e:
        logger.error(f"NutriScan: Trigger classification failed: {str(e)}. Using fallback rules.")
        # Fallback to general heuristics if LLM classification fails
        is_high = nutrition.get("spicy") or nutrition.get("acidic") or nutrition.get("caffeine") or nutrition.get("chocolate") or nutrition.get("fat", 0.0) > 15.0
        trigger_level = "HIGH" if is_high else ("MEDIUM" if nutrition.get("fat", 0.0) >= 10.0 else "LOW")
        reasoning = "Kutipan Medis: Makanan pedas/tinggi lemak dikaitkan dengan relaksasi LES dan iritasi lambung."

    analysis = FoodAnalysis(
        food_name=food_name,
        nutrition_facts=nutrition,
        gerd_trigger_level=trigger_level,
        reasoning=reasoning
    )
    
    logger.info(f"NutriScan: Trigger level resolved to {trigger_level} based on medical RAG evidence.")
    return {"food_analysis": analysis}