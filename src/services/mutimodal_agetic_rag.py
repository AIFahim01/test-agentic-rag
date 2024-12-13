import openai
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.vectordb.chroma import ChromaDb
from pathlib import Path
from dotenv import load_dotenv
import os
import subprocess
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(text, model="text-embedding-ada-002"):
    """Generate embeddings using OpenAI's embedding API."""
    text = text.replace("\n", " ")  # Clean up the text
    try:
        response = openai.Embedding.create(input=[text], model=model)
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def extract_tables_with_nougat(pdf_path, output_dir="output_tables"):
    """Extract tables from the PDF using Nougat OCR."""
    try:
        # Run Nougat command to extract tables
        subprocess.run(
            ["nougat", pdf_path, "-o", output_dir],
            check=True
        )
        # Read the extracted tables as JSON
        tables = []
        for table_file in Path(output_dir).glob("*.json"):
            with open(table_file, "r") as file:
                table_data = json.load(file)
                tables.append(table_data)
        return tables
    except Exception as e:
        print(f"Error extracting tables with Nougat: {e}")
        return []


class MultimodalRAG:
    def __init__(self, pdf_path: str, collection: str, db_path: str):
        self.pdf_path = pdf_path
        self.text_vector_db = ChromaDb(
            collection=f"{collection}_text",
            path=db_path,
            persistent_client=True,
        )
        self.image_vector_db = ChromaDb(
            collection=f"{collection}_images",
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

    def parse_pdf(self):
        """Parse the PDF into text, tables (using Nougat), and images."""
        # Extract tables using Nougat
        nougat_tables = extract_tables_with_nougat(self.pdf_path)

        # Manually categorize elements
        categorized_elements = []

        # Add Nougat tables to categorized elements
        for table in nougat_tables:
            categorized_elements.append({"type": "table", "content": json.dumps(table)})

        return categorized_elements

    def process_elements(self, elements):
        """Process text, tables, and images, storing them in the vector database."""
        for element in elements:
            try:
                if element["type"] == "text":
                    summary = self.agent.print_response(f"Summarize the following text:\n\n{element['content']}",
                                                        stream=False)
                    if summary and isinstance(summary, str) and summary.strip():
                        print(f"Processing text summary: {summary[:50]}...")
                        embedding = get_embedding(summary)
                        if embedding:
                            self.text_vector_db.add(
                                documents=[summary],
                                embeddings=[embedding],
                                ids=[f"text-{hash(summary)}"],
                            )
                        else:
                            print(f"Failed to generate embedding for text: {summary}")
                    else:
                        print("Summary for text element is empty or invalid. Skipping.")

                elif element["type"] == "table":
                    table_content = element.get("content", "").strip()
                    if table_content:
                        print(f"Processing table content: {table_content[:50]}...")
                        embedding = get_embedding(table_content)
                        if embedding:
                            self.text_vector_db.add(
                                documents=[table_content],
                                embeddings=[embedding],
                                ids=[f"table-{hash(table_content)}"],
                            )
                        else:
                            print(f"Failed to generate embedding for table: {table_content}")
                    else:
                        print("Table content is empty or invalid. Skipping.")

                elif element["type"] == "image":
                    image_path = element["content"]
                    description = self.agent.print_response(
                        f"Describe the following image in detail.", images=[image_path], stream=False
                    )
                    if description and isinstance(description, str) and description.strip():
                        print(f"Processing image description: {description[:50]}...")
                        embedding = get_embedding(description)
                        if embedding:
                            self.image_vector_db.add(
                                documents=[description],
                                embeddings=[embedding],
                                ids=[f"image-{hash(description)}"],
                            )
                        else:
                            print(f"Failed to generate embedding for image: {description}")
                    else:
                        print(f"Description for image {image_path} is empty. Skipping.")

            except Exception as e:
                print(f"Error processing element: {element}. Error: {e}")

    def query(self, prompt: str):
        """Query the system to retrieve relevant content and generate a response."""
        text_results = self.text_vector_db.query(query_texts=[prompt], n_results=3)
        image_results = self.image_vector_db.query(query_texts=[prompt], n_results=3)

        context = "### Text Summaries:\n"
        for doc in text_results["documents"]:
            context += f"- {doc}\n"

        context += "\n### Image Descriptions:\n"
        for doc in image_results["documents"]:
            context += f"- {doc}\n"

        final_response = self.agent.print_response(f"Context:\n{context}\n\nQuestion: {prompt}", stream=True)
        return final_response


if __name__ == "__main__":
    pdf_path = "/home/aifahim/PycharmProjects/test-agentic-rag/src/data/pdfs/Otani_Toward_Verifiable_and_Reproducible_Human_Evaluation_for_Text-to-Image_Generation_CVPR_2023_paper.pdf"
    collection = "multimodal_rag_collection"
    db_path = "."

    rag = MultimodalRAG(pdf_path, collection, db_path)
    parsed_elements = rag.parse_pdf()

    tables = [el for el in parsed_elements if el["type"] == "table"]
    if not tables:
        print("No tables found in the parsed PDF.")
    else:
        print("First table content:", tables[0]["content"])

    # Uncomment below lines to process elements or query the system.
    # rag.process_elements(parsed_elements)
    # query = "What are the key insights from the tables and images in the document?"
    # response = rag.query(query)
    # print(response)
