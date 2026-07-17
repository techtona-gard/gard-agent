from pydantic import BaseModel, Field
from src.state import GraphState
from src.config import get_llm
from src.tools.fatsecret_api import fatsecret_tool

class FoodExtraction(BaseModel):
    is_food_query: bool = Field(description="Apakah query mengandung pertanyaan atau pernyataan tentang makanan/minuman/makan")
    food_name: str = Field(default="", description="Nama spesifik makanan atau minuman yang diekstrak, kosong jika tidak ada")

def run_nutri_node(state: GraphState) -> GraphState:
    chat_input = state["chat_input"]
    
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(FoodExtraction)
    
    extraction = structured_llm.invoke(f"Ekstrak informasi makanan dari teks berikut jika ada: {chat_input}")
    
    food_analysis = None
    if extraction.is_food_query and extraction.food_name:
        food_analysis = fatsecret_tool.get_nutrition(extraction.food_name)
        
    return {"food_analysis": food_analysis}
