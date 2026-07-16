import logging
from src.state import GraphState
from src.schemas.payloads import IntentType, OperationMode
from src.config import router_llm, load_prompt
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class RoutePrediction(BaseModel):
    intent: IntentType = Field(description="The routed intent based on user message")

def determine_intent(state: GraphState) -> GraphState:
    """
    Orchestrator node to route the incoming payload and decide operation mode.
    Sets 'operation_mode' and 'intent' in the graph state.
    """
    logger.info("Orchestrator: Categorising mode and routing...")
    payload = state["payload"]
    
    # 1. Determine Operation Mode
    # Use explicitly requested mode if present, else infer
    if payload.operation_mode:
        mode = payload.operation_mode
    elif payload.message or payload.image_url:
        mode = OperationMode.INTERACTIVE
    else:
        mode = OperationMode.TELEMETRY_SYNC
        
    logger.info(f"Orchestrator: Operation mode resolved to: {mode.value}")
    
    # 2. Determine Intent/Route
    if mode == OperationMode.TELEMETRY_SYNC:
        intent = IntentType.PROCESS_LIFESTYLE
    else:
        # Interactive routing
        if payload.image_url:
            intent = IntentType.PROCESS_FOOD
        elif payload.message:
            try:
                # Structured routing via LLM
                prompt_template = load_prompt("orchestrator_prompt.txt")
                system_msg = f"{prompt_template}\n\nUser Message: {payload.message}"
                
                structured_llm = router_llm.with_structured_output(RoutePrediction)
                prediction = structured_llm.invoke(system_msg)
                intent = prediction.intent
            except Exception as e:
                logger.error(f"Orchestrator: LLM routing failed: {str(e)}. Fallback to heuristics.")
                msg_lower = payload.message.lower()
                if any(w in msg_lower for w in ["makan", "minum", "food", "nutrisi", "seblak", "indomie", "kopi"]):
                    intent = IntentType.PROCESS_FOOD
                elif any(w in msg_lower for w in ["tidur", "stres", "jadwal", "kuliah", "sleep"]):
                    intent = IntentType.PROCESS_LIFESTYLE
                else:
                    intent = IntentType.GENERAL_CHAT
        else:
            intent = IntentType.UNKNOWN
            
    logger.info(f"Orchestrator: Route intent resolved to: {intent.value}")
    
    # We update the payload inside state with the operation mode so subsequent agents can read it
    payload.operation_mode = mode
    
    return {
        "intent": intent,
        "payload": payload
    }