import os
import sys
import logging
from dotenv import load_dotenv

# Ensure project root is in import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.rag_service import RAGService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_seed():
    """Seeds the Supabase database with the medical guidelines."""
    load_dotenv()
    logger.info("Initializing RAG Service...")
    rag = RAGService()
    
    logger.info("Starting database seeding process...")
    result = rag.seed_database()
    
    if result["success"]:
        logger.info(f"Database seeding completed successfully! Inserted {result['inserted_count']} documents.")
    else:
        logger.error(f"Database seeding failed: {result['reason']}")
        logger.error("Make sure you have created the table and pgvector extension in Supabase SQL editor.")

if __name__ == "__main__":
    run_seed()
