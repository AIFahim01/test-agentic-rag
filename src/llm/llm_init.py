import os
from config import OPENAI_API_KEY, DEFAULT_LLM_MODEL
from llama_index.llms.openai import OpenAI

def initialize_llm(model_name: str = DEFAULT_LLM_MODEL):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    return OpenAI(model=model_name)
