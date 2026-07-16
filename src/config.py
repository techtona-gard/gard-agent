from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# =====================================================
# Load Environment Variables
# =====================================================
load_dotenv()

# =====================================================
# LLM Configuration
# =====================================================

router_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

nutrition_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

expert_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)