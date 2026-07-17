import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class RAGChroma:
    def __init__(self, persist_directory: str = None):
        # Gunakan env var jika tidak ada, default ke folder lokal 'chroma_db'
        if persist_directory is None:
            self.persist_directory = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db")
        else:
            self.persist_directory = persist_directory
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            if "GOOGLE_API_KEY" not in os.environ:
                os.environ["GOOGLE_API_KEY"] = api_key
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        else:
            self.retriever = None

    def search(self, query: str) -> str:
        if not self.retriever:
            return "RAG tidak aktif karena GOOGLE_API_KEY tidak dikonfigurasi."
            
        try:
            if not os.path.exists(self.persist_directory) or not os.listdir(self.persist_directory):
                return "Database RAG medis masih kosong."
                
            docs = self.retriever.invoke(query)
            if not docs:
                return "Tidak ada informasi relevan dalam database medis."
            
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return f"Error saat mencari di ChromaDB: {e}"

rag_tool = RAGChroma()
