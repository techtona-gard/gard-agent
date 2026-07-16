import logging
from fastapi import FastAPI, HTTPException
from src.schemas.payloads import UserPayload
from src.graph import gard_graph

# Configure standard logging for production readiness
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GARD (GERD Guard) AI Service",
    description="Multi-Agent AI Service (FastAPI + LangGraph) for GERD Risk Prediction.",
    version="1.0.0"
)

@app.get("/")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "GARD AI Service is running."}

@app.post("/api/v1/analyze")
async def analyze_payload(payload: UserPayload):
    """
    Main entry point for incoming JSON payloads from the ExpressJS Backend.
    Triggers the LangGraph multi-agent workflow.
    """
    logger.info(f"Received payload for user_id: {payload.user_id}")
    
    try:
        # Initialize Graph State based on TypedDict schema
        initial_state = {
            "payload": payload,
            "intent": None,
            "food_analysis": None,
            "lifestyle_analysis": None,
            "final_response": None,
            "messages": [],
            "errors": []
        }
        
        # Execute the LangGraph workflow
        result = gard_graph.invoke(initial_state)
        
        return {
            "success": True,
            "data": result.get("final_response")
        }
        
    except Exception as e:
        logger.error(f"Error during LangGraph execution: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during AI analysis")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)