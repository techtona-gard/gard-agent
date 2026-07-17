import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.schemas import AnalyzeRequest
from src.graph import create_graph_app
from dotenv import load_dotenv
from langchain_core.runnables.config import RunnableConfig

load_dotenv()

# Variabel global untuk menampung koneksi DB dan app LangGraph
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
        
        # Inisialisasi Connection Pool ke PostgreSQL (Railway)
        pool = ConnectionPool(
            conninfo=db_url,
            max_size=20,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        
        # Setup table checkpointer di Postgres (otomatis dibuat jika belum ada)
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        
        gard_graph = create_graph_app(checkpointer)
        print("Berhasil terhubung ke PostgreSQL.")
    else:
        print("DATABASE_URL tidak ditemukan. Menggunakan MemorySaver (RAM) lokal sementara...")
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        gard_graph = create_graph_app(checkpointer)
        
    yield
    
    if pool:
        print("Menutup koneksi PostgreSQL...")
        pool.close()

app = FastAPI(title="GARD AI Microservice", version="2.0.0", lifespan=lifespan)

@app.post("/api/v1/analyze")
async def analyze(request: AnalyzeRequest):
    if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        
    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY (atau GEMINI_API_KEY) is not configured.")
        
    if gard_graph is None:
        raise HTTPException(status_code=500, detail="Graph belum terinisialisasi.")
        
    try:
        # State awal
        initial_state = {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
            "baseline_gerd_q": request.baseline_gerd_q,
            "sensor_data": request.sensor_data,
            "chat_input": request.chat_input,
            "food_analysis": None, 
        }
        
        config = RunnableConfig(configurable={"thread_id": request.thread_id})
        
        # Invoke Graph
        result = gard_graph.invoke(initial_state, config=config)
        
        final_response = result.get("final_response")
        if not final_response:
             raise HTTPException(status_code=500, detail="Gagal menghasilkan respons akhir.")

        return {
            "thread_id": request.thread_id,
            "ai_message": final_response.ai_message,
            "meal_schedule": [m.dict() for m in final_response.meal_schedule] if final_response.meal_schedule else None,
            "new_chat_summary": final_response.new_chat_summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
