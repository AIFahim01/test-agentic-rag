def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def get_embeddings(text: str, model_name: str = "text-embedding-ada-002"):
    """
    Given a text input, return the vector embeddings using OpenAI's embedding model.
    """
    response = openai.Embedding.create(
        input=[text],
        model=model_name
    )
    return response['data'][0]['embedding']

def extract_text_from_pdf(pdf_path: str):
    # Implement actual PDF-to-text extraction here.
    # Placeholder implementation:
    return "Full text of PDF."
