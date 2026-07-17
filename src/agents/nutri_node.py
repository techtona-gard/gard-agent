from pydantic import BaseModel, Field
from src.state import GraphState
from src.config import get_llm
from src.tools.fatsecret_api import fatsecret_tool
from src.schemas import ScanFoodResponse
from langchain_core.messages import HumanMessage

class FoodExtraction(BaseModel):
    is_food_query: bool = Field(description="Apakah gambar/teks mengandung makanan atau minuman")
    food_name: str = Field(default="", description="Nama spesifik makanan atau minuman, kosong jika tidak ada")
    ai_message: str = Field(description="Penjelasan singkat mengenai apa yang ada di gambar/teks")

def run_nutri_node(state: GraphState) -> GraphState:
    chat_input = state.get("chat_input") or ""
    image_base64 = state.get("image_base64")
    
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(FoodExtraction)
    
    content = [{"type": "text", "text": f"Ekstrak informasi makanan dari konteks ini. Teks User: {chat_input}"}]
    
    if image_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        })
    
    message = HumanMessage(content=content)
    
    extraction = structured_llm.invoke([message])
    
    food_analysis = None
    if extraction.is_food_query and extraction.food_name:
        food_analysis = fatsecret_tool.get_nutrition(extraction.food_name)
        
    response = ScanFoodResponse(
        ai_message=extraction.ai_message,
        food_profile=food_analysis
    )
        
    return {
        "food_analysis": food_analysis,
        "scan_response": response,
        "chat_summary": state.get("chat_summary", "") + f" User memindai makanan: {extraction.food_name}. AI merespons: {extraction.ai_message}."
    }
