import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.services.fatsecret_service import FatSecretService
from src.services.rag_service import RAGService

logger = logging.getLogger(__name__)

# =====================================================
# Load Environment Variables
# =====================================================
load_dotenv()

# Check for GEMINI API key
gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    logger.warning("Config: GEMINI_API_KEY environment variable is not set. Using placeholder to bypass validation.")
    gemini_api_key = "MOCK_API_KEY_PLACEHOLDER"

# =====================================================
# LLM Configuration
# =====================================================
router_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=gemini_api_key
)

nutrition_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=gemini_api_key
)

expert_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=gemini_api_key
)

# =====================================================
# Services Singletons
# =====================================================
fatsecret_service = FatSecretService()
rag_service = RAGService()

# =====================================================
# Prompt Loader Utility
# =====================================================
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

def load_prompt(filename: str) -> str:
    """Loads a prompt template text file from the prompts folder."""
    path = os.path.join(PROMPTS_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Config: Failed to load prompt file {filename} from {path}: {str(e)}")
        raise FileNotFoundError(f"Required prompt file {filename} not found.")