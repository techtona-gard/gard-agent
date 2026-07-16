from typing import TypedDict, Annotated, Optional, List, Dict, Any
import operator
from src.schemas.payloads import UserPayload, FoodAnalysis, LifestyleAnalysis, IntentType

class GraphState(TypedDict):
    """
    Represents the state of our LangGraph computation.
    """
    payload: UserPayload
    intent: Optional[IntentType]
    food_analysis: Optional[FoodAnalysis]
    lifestyle_analysis: Optional[LifestyleAnalysis]
    final_response: Optional[Dict[str, Any]]
    messages: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]