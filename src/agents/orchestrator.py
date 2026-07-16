from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import GardAgentState
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any

class RoutingDecision(BaseModel):
    route: str = Field(description="Pilih rute: 'PROCESS_FOOD', 'PROCESS_LIFESTYLE', atau 'GENERAL_CHAT'")
    reasoning: str = Field(description="Alasan singkat penentuan rute")

def orchestrator_node(state: GardAgentState) -> Dict[str, Any]:
    print("\n[⚡ Node: Orchestrator Agent] Memeriksa paket data...")
    
    message = state.get("user_message", "")
    if state.get("image_path"):
        return {"input_route": "PROCESS_FOOD"}
        
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(RoutingDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Kamu adalah Router Pintu Gerbang untuk GerdGuard.ai. Pilih PROCESS_FOOD jika membahas makanan/minuman, PROCESS_LIFESTYLE jika mengeluh sakit/stres/kuliah, dan GENERAL_CHAT jika sapaan kasual."),
        ("human", "Pesan: '{user_msg}'")
    ])
    
    decision = (prompt | structured_llm).invoke({"user_msg": message})
    return {"input_route": decision.route}