import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal
from src.graph import gard_brain_app

app = FastAPI(title="Gard Brain - Multi Agent Microservice")

class PlatformPayload(BaseModel):
    user_message: str
    image_path: Optional[str] = None
    baseline_gerdq: Literal[
        "RENDAH",
        "SEDANG",
        "TINGGI"
    ] = "RENDAH"
    vitals_data: Dict[str, Any] = Field(default_factory=dict)
    activity_data: Dict[str, Any] = Field(default_factory=dict)

@app.get("/")
def read_root():
    return {"status": "online", "message": "GerdGuard Brain Multi-Agent Core is running!"}

@app.post("/api/v1/analyze")
async def process_agent(payload: PlatformPayload):
    # Konversi payload masuk menjadi format GraphState LangGraph
    inputs = payload.dict()
    
    # Jalankan eksekusi Multi-Agent System
    output_state = gard_brain_app.invoke(inputs)
    
    # Kembalikan respon akhir ke Express.js
    return {
        "status": "success",
        "risk_score": output_state.get("risk_score", 0),
        "final_response": output_state.get("final_response")
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)