from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from src.state import GraphState
from src.agents.nutri_node import run_nutri_node
from src.agents.expert_node import run_expert_node
from src.config import get_llm

class RouteDecision(BaseModel):
    needs_nutri_node: bool = Field(description="True jika teks mengandung pembicaraan tentang makanan, minuman, bahan masakan, atau jadwal makan")

def route_input(state: GraphState) -> str:
    chat_input = state["chat_input"]
    
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(RouteDecision)
    decision = structured_llm.invoke(f"Apakah teks ini menyebutkan nama makanan, minuman, atau makan? Teks: {chat_input}")
    
    if decision.needs_nutri_node:
        return "nutri_node"
    return "expert_node"

def create_graph_app(checkpointer):
    workflow = StateGraph(GraphState)
    
    # Nodes
    workflow.add_node("nutri_node", run_nutri_node)
    workflow.add_node("expert_node", run_expert_node)
    
    # Conditional Entry Point
    workflow.set_conditional_entry_point(
        route_input,
        {
            "nutri_node": "nutri_node",
            "expert_node": "expert_node"
        }
    )
    
    # Edges
    workflow.add_edge("nutri_node", "expert_node")
    workflow.add_edge("expert_node", END)
    
    # Compile with dynamic memory checkpointer (Postgres/MemorySaver)
    app = workflow.compile(checkpointer=checkpointer)
    
    return app
