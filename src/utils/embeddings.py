import os
import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_embeddings(text: str, model_name: str = "text-embedding-ada-002"):
    """
    Given a text input, return the vector embeddings using OpenAI's embedding model.
    """
    response = openai.Embedding.create(
        input=[text],
        model=model_name
    )
    return response['data'][0]['embedding']
