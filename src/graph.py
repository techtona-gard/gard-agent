from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.agents.nutri_node import run_nutri_node
from src.agents.expert_node import run_expert_node
from src.agents.schedule_node import run_schedule_node

def route_action(state: GraphState) -> str:
    action = state.get("action", "chat")
    if action == "scan":
        return "nutri_node"
    elif action == "schedule":
        return "schedule_node"
    return "expert_node"

def create_graph_app(checkpointer):
    workflow = StateGraph(GraphState)
    
    workflow.add_node("nutri_node", run_nutri_node)
    workflow.add_node("expert_node", run_expert_node)
    workflow.add_node("schedule_node", run_schedule_node)
    
    # Conditional Entry Point based on forced Action Type
    workflow.set_conditional_entry_point(
        route_action,
        {
            "nutri_node": "nutri_node",
            "expert_node": "expert_node",
            "schedule_node": "schedule_node"
        }
    )
    
    workflow.add_edge("nutri_node", END)
    workflow.add_edge("expert_node", END)
    workflow.add_edge("schedule_node", END)
    
    app = workflow.compile(checkpointer=checkpointer)
    return app
