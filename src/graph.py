from langgraph.graph import StateGraph, START, END

from src.state import GardAgentState

from src.agents.orchestrator import orchestrator_node
from src.agents.nutri_scan import nutri_scan_node
from src.agents.bio_rhythm import bio_rhythm_node
from src.agents.gerd_expert import gerd_expert_node

# Inisialisasi Alur Kerja Grafik
workflow = StateGraph(GardAgentState)

# Daftarkan Nodes
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("nutri_scan", nutri_scan_node)
workflow.add_node("bio_rhythm", bio_rhythm_node)
workflow.add_node("gerd_expert", gerd_expert_node)

# Conditional Router Logic
def router_edge(state: GardAgentState) -> str:
    route = state.get("input_route")
    if route == "PROCESS_FOOD": return "goToNutri"
    if route == "PROCESS_LIFESTYLE": return "goToBio"
    return "goToExpert"

workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges("orchestrator", router_edge, {
    "goToNutri": "nutri_scan",
    "goToBio": "bio_rhythm",
    "goToExpert": "gerd_expert"
})

workflow.add_edge("nutri_scan", "gerd_expert")
workflow.add_edge("bio_rhythm", "gerd_expert")
workflow.add_edge("gerd_expert", END)

gard_brain_app = workflow.compile()