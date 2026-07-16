from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.agents.orchestrator import determine_intent
from src.agents.nutri_scan import process_food
from src.agents.bio_rhythm import process_lifestyle
from src.agents.gerd_expert import generate_recommendation
from src.schemas.payloads import IntentType

def route_intent(state: GraphState):
    """
    Conditional edge routing logic.
    Supports parallel branching (Fork) if both food and lifestyle biometrics data are present.
    """
    intent = state.get("intent")
    payload = state.get("payload")
    
    destinations = []
    
    # 1. Route to Food Scanner if food intent is triggered
    if intent == IntentType.PROCESS_FOOD:
        destinations.append("nutri_scan")
        
    # 2. Route to Bio Rhythm if wearable/biometrics/calendar data is present in payload
    if payload and (payload.sleep_data or payload.health_data or payload.calendar_events):
        destinations.append("bio_rhythm")
        
    # 3. Fallback to expert chatbot directly if no food or biometrics are present
    if not destinations:
        destinations.append("gerd_expert")
        
    return destinations

def create_graph():
    """
    Constructs and compiles the Multi-Agent LangGraph workflow for GARD.
    """
    workflow = StateGraph(GraphState)
    
    # 1. Add Nodes
    workflow.add_node("orchestrator", determine_intent)
    workflow.add_node("nutri_scan", process_food)
    workflow.add_node("bio_rhythm", process_lifestyle)
    workflow.add_node("gerd_expert", generate_recommendation)
    
    # 2. Define Entry Point
    workflow.set_entry_point("orchestrator")
    
    # 3. Conditional Routing from Orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        route_intent,
        {
            "nutri_scan": "nutri_scan",
            "bio_rhythm": "bio_rhythm",
            "gerd_expert": "gerd_expert"
        }
    )
    
    # 4. Define Standard Edges (Flow into Expert)
    workflow.add_edge("nutri_scan", "gerd_expert")
    workflow.add_edge("bio_rhythm", "gerd_expert")
    
    # 5. Define Exit
    workflow.add_edge("gerd_expert", END)
    
    return workflow.compile()

# Instantiate the compiled graph so it can be imported in main.py
gard_graph = create_graph()