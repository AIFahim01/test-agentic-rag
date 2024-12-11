from langchain_openai import OpenAI, ChatOpenAI
from services.embeddings_services import EmbeddingService
from pathlib import Path
import os


class RAGService:
    def __init__(self, pdf_path, collection_name, db_path):
        # Try initializing OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key is missing. Set it in the environment variable OPENAI_API_KEY.")

        self.chat_model = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
        )
        self.embedding_service = EmbeddingService(persist_directory=db_path, collection_name=collection_name)
        self.pdf_path = pdf_path

    def index_pdfs(self):
        print(f"Indexing PDFs from {self.pdf_path}...")
        for pdf_file in Path(self.pdf_path).glob("*.pdf"):
            self.embedding_service.process_pdf(str(pdf_file))
        print("Indexing complete.")

    def query(self, user_query):
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.embedding_service.find_relevant_chunks(user_query)
            if not relevant_chunks:
                return "No relevant context found in the knowledge base."

            # Prepare context
            context = "\n\n".join(chunk["chunk_text"] for chunk in relevant_chunks)

            # Prepare the prompt
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
            ]

            # Generate response using ChatOpenAI
            response = self.chat_model.invoke(messages)  # Use invoke instead of __call__
            return response.content  # Access the content property of the AIMessage object
        except Exception as e:
            return f"Error querying the Vanilla RAG: {e}"

