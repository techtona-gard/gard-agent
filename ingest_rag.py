import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def ingest_pdf(pdf_directory: str, persist_directory: str = "chroma_db"):
    if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY (atau GEMINI_API_KEY) tidak dikonfigurasi. Pastikan set di .env")
        return

    documents = []
    
    # Tangani path jika berupa direktori atau file langsung
    if os.path.isdir(pdf_directory):
        for file in os.listdir(pdf_directory):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_directory, file)
                print(f"Memuat dokumen PDF: {pdf_path}")
                try:
                    loader = PyPDFLoader(pdf_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Gagal memuat {pdf_path}: {e}")
    elif os.path.isfile(pdf_directory) and pdf_directory.endswith(".pdf"):
        print(f"Memuat dokumen PDF: {pdf_directory}")
        try:
            loader = PyPDFLoader(pdf_directory)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Gagal memuat {pdf_directory}: {e}")
    else:
        print(f"Path {pdf_directory} bukan direktori atau file PDF yang valid.")
        
    if not documents:
        print("Tidak ada dokumen yang dimuat. Pastikan ada file PDF di direktori target.")
        return
    
    print(f"Memecah {len(documents)} halaman dokumen...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"Dihasilkan {len(splits)} chunks teks.")
    
    print("Membuat embeddings dan menyimpan ke ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    print(f"Berhasil menyimpan {len(splits)} chunks ke ChromaDB di direktori '{persist_directory}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest dokumen PDF medis ke ChromaDB untuk RAG")
    parser.add_argument("--pdf_dir", type=str, default="data/pdf_journals", help="Direktori (atau file tunggal) PDF jurnal medis")
    default_db_dir = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db")
    parser.add_argument("--db_dir", type=str, default=default_db_dir, help="Direktori persistensi ChromaDB")
    
    args = parser.parse_args()
    ingest_pdf(args.pdf_dir, args.db_dir)
