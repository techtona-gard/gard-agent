import os
import sys
import logging

# Ensure project root is in import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.rag_service import RAGService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def ingest_medical_journals(journals_dir: str = "data/journals"):
    """
    Ingests all text and markdown files in the specified directory 
    into the ChromaDB vector database.
    """
    rag = RAGService()
    
    if not os.path.exists(journals_dir):
        os.makedirs(journals_dir, exist_ok=True)
        logger.info(f"Created directory: {journals_dir}. Please place your text (.txt) or markdown (.md) medical papers here.")
        return
        
    files = [f for f in os.listdir(journals_dir) if f.endswith((".txt", ".md"))]
    if not files:
        logger.info(f"No journal files found in {journals_dir}. Ingestion skipped.")
        return
        
    logger.info(f"Found {len(files)} journal file(s) for ingestion.")
    
    for file in files:
        file_path = os.path.join(journals_dir, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                
            if not content:
                continue
                
            # Chunking document by double newlines (paragraphs)
            chunks = [chunk.strip() for chunk in content.split("\n\n") if len(chunk.strip()) > 30]
            
            logger.info(f"Ingesting {file} ({len(chunks)} chunks)...")
            for i, chunk in enumerate(chunks):
                doc_id = f"journal_{file.replace('.', '_')}_chunk_{i}"
                rag.collection.add(
                    ids=[doc_id],
                    documents=[chunk],
                    metadatas=[{"source": file, "chunk_index": i}]
                )
            logger.info(f"Successfully ingested {file}.")
            
        except Exception as e:
            logger.error(f"Failed to ingest file {file}: {str(e)}")

if __name__ == "__main__":
    ingest_medical_journals()
