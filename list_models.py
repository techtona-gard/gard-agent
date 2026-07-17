import os
from dotenv import load_dotenv

load_dotenv()

import google.genai as genai

if "GEMINI_API_KEY" in os.environ:
    api_key = os.environ["GEMINI_API_KEY"]
else:
    api_key = os.environ.get("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(f"Name: {m.name}, Display: {m.display_name}")
