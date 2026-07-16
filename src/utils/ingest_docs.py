import os
import sys
import logging
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Ensure project root is in import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def ingest_medical_journals(journals_dir: str = "data/journals"):
    """
    Reads text/markdown files locally and pushes them with Gemini embeddings
    directly into Supabase pgvector database.
    """
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not supabase_url or not supabase_key or not gemini_key:
        logger.error("Ingestion failed: SUPABASE_URL, SUPABASE_KEY, or GEMINI_API_KEY is missing in your local .env file.")
        return
        
    try:
        # Initialize Supabase Client
        client: Client = create_client(supabase_url, supabase_key)
        
        # Initialize Gemini Embedding Model (768 dimensions)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_key
        )
    except Exception as e:
        logger.error(f"Failed to initialize API clients: {str(e)}")
        return
        
    if not os.path.exists(journals_dir):
        os.makedirs(journals_dir, exist_ok=True)
        logger.info(f"Created directory: {journals_dir}. Place your text (.txt) or markdown (.md) papers here.")
        return
        
    files = [f for f in os.listdir(journals_dir) if f.endswith((".txt", ".md"))]
    if not files:
        logger.info(f"No journal files found in {journals_dir} to ingest.")
        return
        
    logger.info(f"Found {len(files)} file(s) for ingestion.")
    
    for file in files:
        file_path = os.path.join(journals_dir, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                
            if not content:
                continue
                
            # Chunking document by double newlines (paragraphs)
            chunks = [chunk.strip() for chunk in content.split("\n\n") if len(chunk.strip()) > 30]
            logger.info(f"Processing '{file}' - split into {len(chunks)} chunks.")
            
            records = []
            for i, chunk in enumerate(chunks):
                doc_id = f"journal_{file.replace('.', '_')}_chunk_{i}"
                
                # Generate embedding vector
                vector = embeddings.embed_query(chunk)
                
                records.append({
                    "id": doc_id,
                    "content": chunk,
                    "metadata": {"source": file, "chunk_index": i},
                    "embedding": vector
                })
                
            # Batch upsert to Supabase
            if records:
                logger.info(f"Pushing {len(records)} vectors to Supabase table 'medical_guidelines'...")
                client.table("medical_guidelines").upsert(records).execute()
                logger.info(f"Successfully ingested all chunks of '{file}'.")
                
        except Exception as e:
            logger.error(f"Failed to ingest file {file}: {str(e)}")

if __name__ == "__main__":
    ingest_medical_journals()
