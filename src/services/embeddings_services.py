from chromadb import PersistentClient
from phi.embedder.openai import OpenAIEmbedder
from pathlib import Path
from pypdf import PdfReader


class EmbeddingService:
    def __init__(self, persist_directory=".chromadb", collection_name="embeddings"):
        self.client = PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = OpenAIEmbedder()

    def process_pdf(self, pdf_path):
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()

        document_id = Path(pdf_path).stem
        embedding = self.embedder.get_embedding(full_text)

        self.collection.add(
            documents=[full_text],
            embeddings=[embedding],
            ids=[document_id],
        )

    def find_relevant_chunks(self, query, limit=3):
        query_embedding = self.embedder.get_embedding(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=limit)

        return [
            {
                "chunk_text": doc,
                "similarity": 1 - dist,
            }
            for doc, dist in zip(results["documents"][0], results["distances"][0])
        ]
