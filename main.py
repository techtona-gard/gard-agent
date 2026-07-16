import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.schemas.payloads import UserPayload
from src.graph import gard_graph

from src.config import router_llm, fatsecret_service

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

@app.get("/api/v1/test/gemini")
async def test_gemini_api():
    """Tests connection to Gemini API by sending a basic query."""
    try:
        response = router_llm.invoke("Say hello in one word.")
        return {
            "success": True,
            "message": "Gemini API is active and key is valid.",
            "response": response.content.strip()
        }
    except Exception as e:
        logger.error(f"Gemini API test failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Gemini API Connection Failed: {str(e)}"
        )

@app.get("/api/v1/test/fatsecret")
async def test_fatsecret_api():
    """Tests connection to FatSecret API by fetching a token and searching for a sample food."""
    client_id = fatsecret_service.client_id
    client_secret = fatsecret_service.client_secret
    
    if not client_id or not client_secret:
        return {
            "success": False,
            "status": "Inactive (Using fallback mock database)",
            "reason": "FATSECRET_CLIENT_ID or FATSECRET_CLIENT_SECRET environment variables are missing."
        }
        
    try:
        token = fatsecret_service._get_token()
        if not token:
            raise ValueError("Failed to retrieve OAuth2 token. Check your credentials.")
            
        # Perform test query
        nutrition = fatsecret_service.search_food_nutrition("apple")
        return {
            "success": True,
            "status": "Active (Using live API)",
            "message": "OAuth2 authentication and search query succeeded.",
            "test_nutrition_facts": nutrition
        }
    except Exception as e:
        logger.error(f"FatSecret API test failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"FatSecret API Connection Failed: {str(e)}"
        )

class JournalIngestRequest(BaseModel):
    text: str
    source: str

@app.post("/api/v1/journals/ingest")
async def ingest_journal_text(req: JournalIngestRequest):
    """
    Ingests text/markdown medical papers, chunks them, generates 768-dim embeddings
    using Gemini, and pushes them in bulk to the ExpressJS backend API to save.
    """
    logger.info(f"Received ingestion request for source: {req.source}")
    from src.config import rag_service
    
    result = rag_service.ingest_text(req.text, req.source)
    if result.get("success"):
        return {
            "success": True,
            "message": f"Successfully ingested {result.get('inserted_count')} chunks.",
            "chunks_inserted": result.get("inserted_count")
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Ingestion failed: {result.get('reason')}"
        )

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