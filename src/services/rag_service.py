import os
import logging
import requests
from typing import List

logger = logging.getLogger(__name__)

# Medical guidelines for local fallback search (no network needed)
SEED_DOCUMENTS = [
    {
        "id": "doc_sleep_eating",
        "text": "Konsensus klinis ACG (American College of Gastroenterology) 2022 merekomendasikan interval minimal 3 jam antara makan malam dan waktu tidur. Mengonsumsi makanan mendekati waktu tidur terbukti secara klinis meningkatkan frekuensi transient lower esophageal sphincter relaxations (TLESRs) akibat peningkatan volume lambung saat posisi berbaring (recumbency), memicu kejadian refluks nokturnal.",
        "metadata": {"category": "sleep", "source": "ACG Clinical Guideline 2022"}
    },
    {
        "id": "doc_stress_les",
        "text": "Penelitian neurogastroenterologi membuktikan bahwa stres psikologis dan kecemasan meningkatkan hipersensitivitas mukosa esofagus terhadap stimulasi asam lambung. Secara fisiologis, stres merangsang pelepasan hormon stres (seperti Corticotropin-Releasing Factor) yang menurunkan tonus tekanan sfingter esofagus bawah (LES) dan memperlambat laju pengosongan lambung (delayed gastric emptying).",
        "metadata": {"category": "stress", "source": "Journal of Neurogastroenterology and Motility"}
    },
    {
        "id": "doc_trigger_foods",
        "text": "Berdasarkan panduan diet klinis esofagus, makanan tinggi lemak memperlambat sekresi kolesistokinin dan memperlama distensi lambung. Makanan pedas (mengandung kapsaisin) mengiritasi mukosa esofagus secara langsung. Zat asam (citrus, tomat), kafein (kopi/teh), dan cokelat bekerja secara farmakologis menurunkan tekanan basal Lower Esophageal Sphincter (LES), yang mempermudah asam lambung naik.",
        "metadata": {"category": "diet", "source": "Clinical Gastroenterology and Hepatology"}
    },
    {
        "id": "doc_student_habits",
        "text": "Studi epidemiologi pada populasi mahasiswa menunjukkan korelasi kuat antara GERD dengan gaya hidup akademik: pola makan tidak teratur, konsumsi kafein dosis tinggi saat begadang, konsumsi mie instan pedas, serta gangguan kualitas tidur sirkadian yang memicu ketidakseimbangan sistem saraf otonom lambung.",
        "metadata": {"category": "student", "source": "BMC Gastroenterology Journal"}
    },
    {
        "id": "doc_remedies",
        "text": "Pedoman tatalaksana non-farmakologis GERD menyarankan elevasi kepala tempat tidur setinggi 15-20 cm (menggunakan gravitasi untuk menahan asam selama berbaring), penurunan berat badan, menghindari berbaring dalam waktu 2-3 jam setelah makan, serta membatasi pakaian ketat yang meningkatkan tekanan intra-abdomen.",
        "metadata": {"category": "remedy", "source": "ACG Clinical Guideline 2022"}
    },
    {
        "id": "doc_gerdq_info",
        "text": "Berdasarkan studi validasi kuesioner GerdQ (Jones et al., 2010), skor total 8-18 memiliki tingkat sensitivitas 65% dan spesifisitas 71% untuk mendiagnosis GERD (setara dengan akurasi diagnosis dokter spesialis). Skor 0-7 menunjukkan probabilitas menderita GERD rendah (22%), sedangkan skor >= 8 mengindikasikan probabilitas tinggi menderita GERD (80%).",
        "metadata": {"category": "gerdq", "source": "Jones et al., GerdQ Validation Study 2010"}
    }
]

from langchain_google_genai import GoogleGenerativeAIEmbeddings

class RAGService:
    def __init__(self):
        self.backend_url = os.getenv("BACKEND_API_URL")
        # Path endpoint di BE untuk query RAG (contoh: /api/v1/ai/medical-guidelines)
        self.endpoint_path = "/api/v1/ai/medical-guidelines"
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.embeddings = None
        self.initialized = True if self.backend_url else False
        
        if self.gemini_key and self.gemini_key != "MOCK_API_KEY_PLACEHOLDER":
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=self.gemini_key
                )
                logger.info("RAG Service: Google Embeddings model initialized.")
            except Exception as e:
                logger.error(f"RAG Service: Failed to initialize Google Embeddings: {str(e)}")
        
        if self.initialized:
            logger.info(f"RAG Service: Configured to pull medical guidelines via BE API at {self.backend_url}")
        else:
            logger.warning("RAG Service: BACKEND_API_URL missing. Fallback local search will be used.")

    def query_context(self, query_text: str, n_results: int = 2) -> str:
        """
        Queries the ExpressJS backend API to fetch medical guidelines RAG context.
        Falls back to local keyword search if ExpressJS backend is unreachable or not configured.
        """
        logger.info(f"RAG Service: Querying medical context for: '{query_text}'")
        
        if self.initialized and self.backend_url:
            try:
                url = f"{self.backend_url.rstrip('/')}{self.endpoint_path}"
                params = {"query": query_text, "limit": n_results}
                
                logger.info(f"RAG Service: Sending GET request to {url}")
                response = requests.get(url, params=params, timeout=3)
                
                if response.status_code == 200:
                    data = response.json()
                    # Mengharapkan response JSON list of strings: ["teks 1", "teks 2"]
                    if isinstance(data, list) and data:
                        logger.info(f"RAG Service: Successfully retrieved {len(data)} context items from BE API.")
                        return "\n---\n".join(data)
                else:
                    logger.warning(f"RAG Service: BE API returned status {response.status_code}. Using fallback.")
            except Exception as e:
                logger.error(f"RAG Service: Failed to query BE API: {str(e)}. Falling back to local search.")

        # Local Keyword-based Fallback Search (Medical RAG Grounded - No own rules)
        logger.info("RAG Service: Executing fallback local keyword search...")
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
            
        return "\n---\n".join([SEED_DOCUMENTS[2]["text"], SEED_DOCUMENTS[4]["text"]])

    def ingest_text(self, text: str, source: str) -> dict:
        """
        Chunks the input text, generates Gemini embeddings, and sends
        the vectors + metadata in bulk to the ExpressJS backend API to save.
        """
        if not self.backend_url:
            return {"success": False, "reason": "BACKEND_API_URL is not set in environment."}
            
        if not self.embeddings:
            return {"success": False, "reason": "Gemini Embeddings not initialized (check API key)."}
            
        try:
            # Chunking document by double newlines (paragraphs)
            chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 30]
            if not chunks:
                return {"success": False, "reason": "No text paragraphs longer than 30 characters found."}
                
            records = []
            for i, chunk in enumerate(chunks):
                doc_id = f"journal_{source.replace('.', '_')}_chunk_{i}"
                vector = self.embeddings.embed_query(chunk)
                records.append({
                    "id": doc_id,
                    "content": chunk,
                    "metadata": {"source": source, "chunk_index": i},
                    "embedding": vector
                })
                
            # Send bulk records to ExpressJS backend API
            url = f"{self.backend_url.rstrip('/')}{self.endpoint_path}/bulk"
            logger.info(f"RAG Service: Sending {len(records)} embedded chunks to backend bulk insert endpoint: {url}")
            
            response = requests.post(url, json={"records": records}, timeout=10)
            if response.status_code in [200, 201]:
                return {"success": True, "inserted_count": len(records)}
            else:
                return {
                    "success": False, 
                    "reason": f"Backend returned status code {response.status_code}: {response.text}"
                }
        except Exception as e:
            logger.error(f"RAG Service: Dynamic ingestion failed: {str(e)}")
            return {"success": False, "reason": str(e)}
