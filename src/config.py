import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"

PDFS = [
    "src/data/pdfs/Albert Einstein book.pdf",
    "src/data/pdfs/Einstein_Relativity.pdf",
    "src/data/pdfs/ingelesa.pdf"
]

SIMILARITY_TOP_K = 3
