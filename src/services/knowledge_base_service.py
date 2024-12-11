from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.chroma import ChromaDb
import os


class KnowledgeBaseService:
    def __init__(self, pdf_path: str, collection: str, db_path: str, persistent: bool = True):
        """
        Initialize the Knowledge Base Service.

        :param pdf_path: Path to the directory containing PDF files.
        :param collection: Name of the ChromaDB collection.
        :param db_path: Path to the ChromaDB data directory.
        :param persistent: Whether to enable persistent storage for ChromaDB.
        """
        self.pdf_path = pdf_path
        self.vector_db = ChromaDb(
            collection=collection,
            path=db_path,
            persistent_client=persistent,
        )
        self.knowledge_base = PDFKnowledgeBase(path=pdf_path, vector_db=self.vector_db)

    def load_knowledge_base(self, recreate: bool = False):
        """
        Load PDFs into the knowledge base.

        :param recreate: Whether to recreate the vector database.
        """
        if not os.path.exists(self.pdf_path) or not os.listdir(self.pdf_path):
            print(f"Error: No PDF files found in the directory {self.pdf_path}.")
            return

        print(f"Loading PDFs from {self.pdf_path} into the knowledge base...")
        self.knowledge_base.load(recreate=recreate)
        print("Knowledge base loaded successfully.")

    def search(self, query: str, num_documents: int = 5):
        """
        Search the knowledge base for relevant documents.

        :param query: The search query.
        :param num_documents: Number of documents to retrieve.
        :return: Search results from the knowledge base.
        """
        try:
            results = self.knowledge_base.search(query=query, num_documents=num_documents)
            if not results:
                print("No relevant results found.")
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []
