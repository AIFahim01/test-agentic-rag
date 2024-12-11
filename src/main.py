from dotenv import load_dotenv
from services.agentic_rag_service import LLMRAgentService
from services.vanila_rag_service import RAGService
from rich.prompt import Prompt
import os


def main():
    # Load environment variables
    load_dotenv()

    # Configuration from environment variables
    PDF_PATH = os.getenv("PDF_PATH", "src/data/pdfs")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_documents")
    DB_PATH = os.getenv("DB_PATH", ".chromadb")
    USER = os.getenv("USER", "user")  # Default user

    # Initialize the Agentic RAG service
    agentic_rag_service = LLMRAgentService(
        pdf_path=PDF_PATH,
        collection=COLLECTION_NAME,
        db_path=DB_PATH,
        enable_rag=True,
        persistent=True,
    )

    # Initialize the Vanilla RAG service
    vanilla_rag_service = RAGService(
        pdf_path=PDF_PATH,
        collection_name=COLLECTION_NAME,
        db_path=DB_PATH,
    )

    # Index PDFs for Vanilla RAG
    print("Indexing PDFs for Vanilla RAG...")
    vanilla_rag_service.index_pdfs()

    print(f"Interactive session started for {USER}.")
    print("Type 'exit' to quit.")

    while True:
        # Get user input
        message = Prompt.ask(f"{USER}: ")
        if message.lower() in ("exit", "bye"):
            print("Goodbye!")
            break

        # Agentic RAG response
        print("\n[Agentic RAG Response]")
        try:
            agentic_response = agentic_rag_service.agent.run(message)
            print(agentic_response.content)
        except Exception as e:
            print(f"Error in Agentic RAG: {e}")

        # Vanilla RAG response
        print("\n[Vanilla RAG Response]")
        try:
            vanilla_response = vanilla_rag_service.query(message)
            print(vanilla_response)
        except Exception as e:
            print(f"Error in Vanilla RAG: {e}")


if __name__ == "__main__":
    main()
