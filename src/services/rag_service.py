import os
import logging
from typing import List
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Medical guidelines to seed ChromaDB for GERD context
SEED_DOCUMENTS = [
    {
        "id": "doc_sleep_eating",
        "text": "Hindari makan dalam waktu 3 jam sebelum tidur. Tidur dengan perut penuh dapat meningkatkan tekanan intra-abdomen, melemahkan LES, dan menyebabkan asam lambung naik kembali ke esofagus ketika berbaring.",
        "metadata": {"category": "sleep"}
    },
    {
        "id": "doc_stress_les",
        "text": "Stres dan kecemasan tinggi memicu pelepasan hormon yang dapat memperlambat pencernaan dan mengendurkan Lower Esophageal Sphincter (LES), yang mempermudah terjadinya GERD (Refluks Asam).",
        "metadata": {"category": "stress"}
    },
    {
        "id": "doc_trigger_foods",
        "text": "Makanan tinggi lemak, pedas, asam (seperti buah jeruk dan tomat), cokelat, mint, kafein (kopi/teh), dan minuman berkarbonasi adalah pemicu utama GERD yang melemaskan LES atau meningkatkan sekresi asam lambung.",
        "metadata": {"category": "diet"}
    },
    {
        "id": "doc_student_habits",
        "text": "Mahasiswa sering mengalami GERD akibat pola makan tidak teratur, konsumsi kopi berlebih saat begadang belajar, konsumsi mie instan pedas, serta stres akademis (jadwal kuliah padat dan kurang tidur).",
        "metadata": {"category": "student"}
    },
    {
        "id": "doc_remedies",
        "text": "Untuk meredakan GERD secara cepat, tidurlah dengan posisi kepala lebih tinggi (elevasi kepala 15-20 cm), longgarkan pakaian ketat di area perut, hindari berbaring setelah makan, serta konsumsi antasida jika diperlukan.",
        "metadata": {"category": "remedy"}
    }
]

class RAGService:
    def __init__(self, persist_directory: str = "data/chroma"):
        self.persist_directory = persist_directory
        self.collection_name = "gerd_guidelines"
        self.client = None
        self.collection = None
        self.initialized = False
        self._init_db()

    def _init_db(self):
        """Initializes ChromaDB and seeds documents if empty."""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(self.persist_directory), exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Using default embedding function (requires sentence-transformers)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            
            # Seed documents if the collection is empty
            if self.collection.count() == 0:
                logger.info("RAG Service: Seeding ChromaDB with medical guidelines...")
                ids = [doc["id"] for doc in SEED_DOCUMENTS]
                documents = [doc["text"] for doc in SEED_DOCUMENTS]
                metadatas = [doc["metadata"] for doc in SEED_DOCUMENTS]
                
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info("RAG Service: Seeding completed.")
            else:
                logger.info(f"RAG Service: Collection already has {self.collection.count()} documents.")
                
            self.initialized = True
        except Exception as e:
            logger.error(f"RAG Service: Failed to initialize ChromaDB. Fallback to basic string search will be used. Error: {str(e)}")
            self.initialized = False

    def query_context(self, query_text: str, n_results: int = 2) -> str:
        """
        Queries ChromaDB for medical context.
        Falls back to keyword matching if ChromaDB is not initialized or fails.
        """
        logger.info(f"RAG Service: Querying medical guidelines for '{query_text}'")
        
        if self.initialized and self.collection:
            try:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results
                )
                documents = results.get("documents", [[]])[0]
                if documents:
                    logger.info("RAG Service: Retrieved context from ChromaDB.")
                    return "\n---\n".join(documents)
            except Exception as e:
                logger.error(f"RAG Service: Query failed, falling back to manual search: {str(e)}")
                
        # Keyword-based Fallback Search
        logger.info("RAG Service: Executing fallback keyword search...")
        matched_docs = []
        query_words = query_text.lower().split()
        
        for doc in SEED_DOCUMENTS:
            doc_text = doc["text"].lower()
            score = sum(1 for word in query_words if word in doc_text)
            if score > 0:
                matched_docs.append((score, doc["text"]))
                
        # Sort by matched word count descending
        matched_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc[1] for doc in matched_docs[:n_results]]
        
        if top_docs:
            return "\n---\n".join(top_docs)
            
        # Absolute fallback: Return the most general docs
        return "\n---\n".join([SEED_DOCUMENTS[2]["text"], SEED_DOCUMENTS[3]["text"]])
