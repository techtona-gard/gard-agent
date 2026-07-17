import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.schemas import ChatRequest, ScanFoodRequest, ScheduleRequest
from src.graph import create_graph_app
from dotenv import load_dotenv
from langchain_core.runnables.config import RunnableConfig

load_dotenv()

pool = None
gard_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, gard_graph
    
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        print("Mendeteksi DATABASE_URL. Menghubungkan ke PostgreSQL untuk persistent memory...")
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver
        
        pool = ConnectionPool(
            conninfo=db_url,
            max_size=20,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        gard_graph = create_graph_app(checkpointer)
    else:
        print("DATABASE_URL tidak ditemukan. Menggunakan MemorySaver (RAM) lokal sementara...")
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        gard_graph = create_graph_app(checkpointer)
        
    yield
    if pool:
        pool.close()

app = FastAPI(title="GARD AI Microservice - Multi Feature", version="3.0.0", lifespan=lifespan)

def ensure_graph():
    if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not configured.")
    if gard_graph is None:
        raise HTTPException(status_code=500, detail="Graph belum terinisialisasi.")

@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest):
    ensure_graph()
    try:
        initial_state = {
            "action": "chat",
            "thread_id": request.thread_id,
            "user_id": request.user_id,
            "baseline_gerd_q": request.baseline_gerd_q,
            "sensor_data": request.sensor_data,
            "chat_input": request.chat_input,
        }
        config = RunnableConfig(configurable={"thread_id": request.thread_id})
        result = gard_graph.invoke(initial_state, config=config)
        return result.get("chat_response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/scan-food")
async def scan_endpoint(request: ScanFoodRequest):
    ensure_graph()
    try:
        initial_state = {
            "action": "scan",
            "thread_id": request.thread_id,
            "user_id": request.user_id,
            "image_base64": request.image_base64,
            "chat_input": request.chat_input,
        }
        config = RunnableConfig(configurable={"thread_id": request.thread_id})
        result = gard_graph.invoke(initial_state, config=config)
        return result.get("scan_response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/schedule")
async def schedule_endpoint(request: ScheduleRequest):
    ensure_graph()
    try:
        initial_state = {
            "action": "schedule",
            "thread_id": request.thread_id,
            "user_id": request.user_id,
            "baseline_gerd_q": request.baseline_gerd_q,
            "sensor_data": request.sensor_data,
            "date": request.date,
        }
        config = RunnableConfig(configurable={"thread_id": request.thread_id})
        result = gard_graph.invoke(initial_state, config=config)
        return result.get("schedule_response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
