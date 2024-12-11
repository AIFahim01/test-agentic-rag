from phi.vectordb.chroma import ChromaDb
from phi.embedder.openai import OpenAIEmbedder
from pathlib import Path
from pypdf import PdfReader

class IndexingService:
    def __init__(self, chroma_path=".chromadb", collection_name="pdf_documents"):
        self.vector_db = ChromaDb(collection=collection_name, path=chroma_path)
        self.embedder = OpenAIEmbedder()

    def process_pdf(self, pdf_path):
        """
        Extract text from PDF and index it.
        :param pdf_path: Path to the PDF file.
        """
        reader = PdfReader(pdf_path)
        document_id = Path(pdf_path).stem

        for page in reader.pages:
            text = page.extract_text()
            embedding = self.embedder.get_embedding(text)

            self.vector_db.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[{"document_id": document_id}]
            )
