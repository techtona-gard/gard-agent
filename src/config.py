import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm(temperature: float = 0.2):
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        google_api_key=api_key
    )
