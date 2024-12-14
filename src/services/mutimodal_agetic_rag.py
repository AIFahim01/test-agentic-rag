import openai
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.vectordb.chroma import ChromaDb
from phi.embedder.openai import OpenAIEmbedder
from gmft.auto import AutoTableDetector
from gmft.pdf_bindings import PyPDFium2Document
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_tables_with_gmft(pdf_path, output_folder):
    """Extract tables from the PDF using GMFT and save them as images."""
    detector = AutoTableDetector()
    tables = []

    try:
        # Use PyPDFium2 to load and parse the PDF
        doc = PyPDFium2Document(pdf_path)
        for page_number, page in enumerate(doc, start=1):
            detected_tables = detector.extract(page)
            for idx, table in enumerate(detected_tables, start=1):
                # Save the table as an image
                table_image = table.image()
                image_filename = f"table_page{page_number}_table{idx}.png"
                image_path = os.path.join(output_folder, image_filename)
                table_image.save(image_path)
                print(f"Saved table from page {page_number} as image: {image_path}")
                tables.append({
                    "page": page_number,
                    "index": idx,
                    "image_path": image_path
                })
        doc.close()  # Clean up the document
    except Exception as e:
        print(f"Error extracting tables with GMFT: {e}")
        return []

    return tables


class MultimodalRAG:
    def __init__(self, pdf_path: str, collection: str, db_path: str):
        self.pdf_path = pdf_path
        self.embedder = OpenAIEmbedder()  # Use OpenAIEmbedder for embeddings
        self.text_vector_db = ChromaDb(
            collection=f"{collection}_text",
            embedder=self.embedder,
            path=db_path,
            persistent_client=True,
        )
        self.image_vector_db = ChromaDb(
            collection=f"{collection}_images",
            embedder=self.embedder,
            path=db_path,
            persistent_client=True,
        )
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
            instructions=[
                "You are an assistant that processes both text and images.",
                "For text, summarize its content concisely.",
                "For images, provide a detailed description.",
            ],
            markdown=True,
        )

    def parse_pdf(self, table_output_folder):
        """Parse the PDF and save tables as images."""
        os.makedirs(table_output_folder, exist_ok=True)

        # Extract tables using GMFT and save them as images
        tables = extract_tables_with_gmft(self.pdf_path, table_output_folder)

        categorized_elements = []

        # Add GMFT tables to categorized elements
        for table in tables:
            categorized_elements.append({
                "type": "table",
                "content": f"Table on page {table['page']}, index {table['index']}",
                "image_path": table["image_path"]
            })

        return categorized_elements

    def process_elements(self, elements):
        """Process text, tables, and images, storing them in the vector database."""
        for element in elements:
            try:
                if element["type"] == "table":
                    image_path = element.get("image_path", "").strip()
                    if image_path:
                        print(f"Processing table image at: {image_path}")
                        embedding = self.embedder.get_embedding(f"Table image saved at {image_path}")
                        if embedding:
                            self.image_vector_db.add(
                                documents=[f"Table saved at {image_path}"],
                                embeddings=[embedding],
                                ids=[f"table-{hash(image_path)}"],
                            )
                        else:
                            print(f"Failed to generate embedding for table image at: {image_path}")
                    else:
                        print("Table image path is empty or invalid. Skipping.")

            except Exception as e:
                print(f"Error processing element: {element}. Error: {e}")

    def query(self, prompt: str):
        """Query the system to retrieve relevant content and generate a response."""
        try:
            # Embed the prompt to create the query vector
            query_embedding = self.embedder.get_embedding(prompt)

            # Query text vector database
            text_results = self.text_vector_db.get_nearest_neighbors(
                query_embedding=query_embedding,
                n_results=3
            )

            # Query image vector database
            image_results = self.image_vector_db.get_nearest_neighbors(
                query_embedding=query_embedding,
                n_results=3
            )

            # Combine results into a readable context
            context = "### Text Summaries:\n"
            for doc, distance in text_results:
                context += f"- {doc} (Distance: {distance})\n"

            context += "\n### Image Descriptions:\n"
            for doc, distance in image_results:
                context += f"- {doc} (Distance: {distance})\n"

            # Generate a response using GPT-4o
            final_response = self.agent.print_response(f"Context:\n{context}\n\nQuestion: {prompt}", stream=True)
            return final_response

        except Exception as e:
            print(f"Error during query: {e}")
            return "An error occurred while querying the knowledge base."


if __name__ == "__main__":
    pdf_path = "/home/aifahim/PycharmProjects/test-agentic-rag/src/data/pdfs/Otani_Toward_Verifiable_and_Reproducible_Human_Evaluation_for_Text-to-Image_Generation_CVPR_2023_paper.pdf"
    collection = "multimodal_rag_collection"
    db_path = "./knowledge_base"
    table_output_folder = "./extracted_tables"

    rag = MultimodalRAG(pdf_path, collection, db_path)
    parsed_elements = rag.parse_pdf(table_output_folder)
    rag.process_elements(parsed_elements)

    query = "What insights can we gain from the document?"
    response = rag.query(query)
    print(response)
