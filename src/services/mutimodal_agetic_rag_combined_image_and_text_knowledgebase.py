import openai
import os
from pathlib import Path
from dotenv import load_dotenv

from phi.agent import Agent, RunResponse
from phi.model.openai import OpenAIChat
from phi.knowledge.langchain import LangChainKnowledgeBase

from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
# from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from gmft.auto import AutoTableDetector
from gmft.pdf_bindings import PyPDFium2Document
import pypdfium2 as pdfium

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_tables_with_gmft(pdf_path, output_folder):
    detector = AutoTableDetector()
    tables = []
    try:
        doc = PyPDFium2Document(pdf_path)
        for page_number, page in enumerate(doc, start=1):
            detected_tables = detector.extract(page)
            for idx, table in enumerate(detected_tables, start=1):
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
        doc.close()
    except Exception as e:
        print(f"Error extracting tables with GMFT: {e}")
        return []
    return tables

def extract_pdf_text_with_pdfium(pdf_path):
    all_text = []
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            page_text = textpage.get_text_range()
            if page_text and page_text.strip():
                all_text.append(page_text)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return "\n".join(all_text)

class MultimodalRAG:
    def __init__(self, pdf_path: str, collection: str, db_path: str):
        self.pdf_path = pdf_path
        self.collection = collection
        self.db_path = db_path

        # Use OpenAIEmbeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.vectorstore = None
        self.knowledge_base = None

        # Agent for text-based Q&A
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
            instructions=[
                "You are an assistant that processes both text and images.",
                "For text, summarize its content concisely.",
                "For images, provide a detailed description.",
            ],
            markdown=True,
        )

        # Vision-capable agent (Placeholder GPT-4 Vision)
        self.vision_agent = Agent(
            model=OpenAIChat(id="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
            markdown=True,
            instructions=[
                "You are a vision-capable assistant. You will be given an image and asked to create a textual description about it.",
                "Respond with a detailed textual interpretation of the image's content.",
            ]
        )

    def extract_pdf_text(self):
        return extract_pdf_text_with_pdfium(self.pdf_path)

    def parse_pdf(self, table_output_folder):
        os.makedirs(table_output_folder, exist_ok=True)
        tables = extract_tables_with_gmft(self.pdf_path, table_output_folder)
        categorized_elements = []
        for table in tables:
            categorized_elements.append({
                "type": "table",
                "content": f"Table on page {table['page']}, index {table['index']}",
                "image_path": table["image_path"]
            })
        return categorized_elements

    def build_vectorstore(self, text_docs):
        if not text_docs.strip():
            print("No text found. Creating a minimal vectorstore with a dummy doc.")
            text_docs = "This document contains no textual content. Only tables or images."

        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # chunks = text_splitter.split_text(text_docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text_docs)

        documents = [Document(page_content=chunk) for chunk in chunks]

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )

    def add_tables_to_vectorstore(self, tables):
        if not self.vectorstore:
            print("Vector store not initialized. Please build it first.")
            return

        table_docs = []
        for table in tables:
            image_path = table.get("image_path", "").strip()
            if image_path:
                image_response = self.vision_agent.run(
                    "Describe this table image in detail.",
                    images=[image_path]
                )
                image_description = image_response.content  # Use .content
                # print(image_description)
                if image_description and image_description.strip():
                    table_docs.append(Document(page_content=image_description))
                else:
                    print(f"No description returned for {image_path}. Skipping.")
            else:
                print("Table image path is empty or invalid. Skipping.")

        if table_docs:
            self.vectorstore.add_documents(table_docs)

    def initialize_knowledge_base(self):
        if not self.vectorstore:
            print("Vector store not created yet. Cannot initialize knowledge base.")
            return
        retriever = self.vectorstore.as_retriever()
        self.knowledge_base = LangChainKnowledgeBase(retriever=retriever)
        self.agent.knowledge_base = self.knowledge_base

    def query(self, prompt: str):
        if not self.knowledge_base:
            print("Knowledge base not initialized.")
            return "Knowledge base not initialized."
        try:
            response = self.agent.run(prompt)
            return response.content  # Use .content instead of .text
        except Exception as e:
            print(f"Error during query: {e}")
            return "An error occurred while querying the knowledge base."

if __name__ == "__main__":
    pdf_path = "/home/aifahim/PycharmProjects/test-agentic-rag/src/data/pdfs/Otani_Toward_Verifiable_and_Reproducible_Human_Evaluation_for_Text-to-Image_Generation_CVPR_2023_paper.pdf"
    collection = "multimodal_rag_collection"
    db_path = "./knowledge_base"
    table_output_folder = "./extracted_tables"

    rag = MultimodalRAG(pdf_path, collection, db_path)

    # Extract text from PDF
    pdf_text = rag.extract_pdf_text()

    # Build vectorstore from extracted text
    rag.build_vectorstore(pdf_text)

    # Extract tables as images
    parsed_elements = rag.parse_pdf(table_output_folder)

    # Describe table images and add their descriptions to the vector store
    rag.add_tables_to_vectorstore(parsed_elements)

    # Initialize the knowledge base
    rag.initialize_knowledge_base()

    # Query the knowledge base
    query = "give me the annotator performance camparisions for fidality IIA Alignment IAA and Med Time as table"
    response_text = rag.query(query)
    print("Agent Response:\n", response_text)
