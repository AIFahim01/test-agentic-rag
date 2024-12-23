import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

import openai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from unstructured.partition.pdf import partition_pdf
from langchain_community.vectorstores.utils import filter_complex_metadata

from phi.agent import Agent
from phi.model.openai import OpenAIChat
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class ContentElement:
    """Simple class to hold content with metadata."""

    def __init__(self, content: str, content_type: str, metadata: Dict = None):
        self.content = content
        self.content_type = content_type
        self.metadata = metadata or {
            'content_type': content_type,
            'length': len(content),
            'timestamp': datetime.now().isoformat()
        }


class MultimodalRAG:
    """Simplified Multimodal RAG System."""

    def __init__(self, pdf_path: str, output_dir: str):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "images"
        self.kb_dir = self.output_dir / "knowledge_base"

        # Create directories
        for dir_path in [self.output_dir, self.image_dir, self.kb_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'rag.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        # Initialize stores
        self.text_store = None
        self.image_store = None
        self.table_store = None

        # Initialize agents
        self._setup_agents()

    def _setup_agents(self):
        """Initialize AI agents."""
        # Text analysis agent
        self.text_agent = Agent(
            name="Text Analyzer",
            role="Process text content",
            model=OpenAIChat(id="gpt-4"),
            instructions=["Analyze and extract key information from text."],
            markdown=True
        )

        # Vision analysis agent
        self.vision_agent = Agent(
            name="Vision Analyzer",
            role="Process visual content",
            model=OpenAIChat(id="gpt-4o"),
            instructions=["Analyze images and visual content in detail."],
            markdown=True
        )

        # Integration agent
        self.integration_agent = Agent(
            name="Integration Specialist",
            role="Combine insights",
            model=OpenAIChat(id="gpt-4"),
            team=[self.text_agent, self.vision_agent],
            instructions=["Combine insights from text and visual content."],
            markdown=True
        )

    def _clean_metadata(self, metadata: Dict) -> Dict:
        """Clean and simplify metadata for Chroma storage."""
        if not metadata:
            return {}

        cleaned = {}
        try:
            for key, value in metadata.items():
                # Convert complex structures to JSON strings
                if isinstance(value, (dict, list, tuple)):
                    cleaned[key] = json.dumps(value)
                # Keep simple types as is
                elif isinstance(value, (str, int, float, bool)):
                    cleaned[key] = value
                # Convert other types to strings
                else:
                    cleaned[key] = str(value)

            return cleaned
        except Exception as e:
            self.logger.warning(f"Error cleaning metadata: {str(e)}")
            return {'error': 'metadata cleaning failed'}

    # @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _process_with_retry(self, agent: Agent, prompt: str, images: List[str] = None) -> str:
        """Process content with retry logic."""
        try:
            response = agent.run(prompt, images=images) if images else agent.run(prompt)
            return response.content
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            raise

    def _extract_text_content(self):
        """Extract text content from PDF."""
        try:
            self.logger.info("Extracting text content...")
            text_elements = partition_pdf(
                filename=self.pdf_path,
                include_metadata=True,
                strategy="hi_res",
                chunking_strategy="by_title",
                max_characters=3000,
                combine_text_under_n_chars=200,
                multipage_sections=True,
                extract_images_in_pdf=False
            )

            valid_elements = []
            for element in text_elements:
                element_dict = element.to_dict() if hasattr(element, 'to_dict') else {}
                category = element_dict.get('category', '')

                if category not in ["Header", "Footer", "Image", "Table"]:
                    content = element_dict.get('text', '')
                    metadata = {
                        'page_number': element_dict.get('metadata', {}).get('page_number'),
                        'category': category,
                        'element_type': element_dict.get('type', 'text')
                    }
                    valid_elements.append(ContentElement(content, "text", metadata))

            return valid_elements

        except Exception as e:
            self.logger.error(f"Text extraction error: {str(e)}")
            raise

    def _extract_image_content(self):
        """Extract image content from PDF with flexible file matching."""
        try:
            self.logger.info("Extracting image content...")
            image_elements = partition_pdf(
                filename=self.pdf_path,
                include_metadata=True,
                strategy="hi_res",
                extract_images_in_pdf=True,
                extract_image_block_output_dir=str(self.image_dir),
                extract_image_block_types=["Image", "Figure"],
                include_image_data=True
            )

            # Get all image files
            image_files = list(self.image_dir.glob("*.*"))
            self.logger.info(f"Found {len(image_files)} image files in output directory")

            # Debug log all files
            for img_file in image_files:
                self.logger.debug(f"Available image file: {img_file.name}")

            # Track processed files
            processed_files = set()
            valid_elements = []

            for element in image_elements:
                element_dict = element.to_dict() if hasattr(element, 'to_dict') else {}
                element_type = element_dict.get('type', '')
                category = element_dict.get('category', '')

                if category == "Image" or element_type == "Image" or "Figure" in str(element_type):
                    metadata = element_dict.get('metadata', {})
                    page_num = metadata.get('page_number')
                    coordinates = metadata.get('coordinates', {})
                    element_id = metadata.get('element_id', '')

                    if page_num:
                        # Find any available image file that hasn't been processed
                        for img_file in image_files:
                            file_path = str(img_file)
                            if file_path in processed_files:
                                continue

                            # Add to valid elements
                            processed_files.add(file_path)

                            # Get figure number from file name if possible
                            fig_num = None
                            filename = img_file.name
                            if "figure" in filename.lower():
                                parts = filename.replace('.jpg', '').replace('.png', '').split('-')
                                if len(parts) >= 2:
                                    try:
                                        fig_num = parts[1]
                                    except:
                                        pass

                            content = element_dict.get('text', '')
                            clean_metadata = {
                                'image_path': file_path,
                                'page_number': page_num,
                                'figure_number': fig_num,
                                'element_id': element_id,
                                'content_type': 'image',
                                'filename': filename,
                                'coordinates': json.dumps(coordinates) if coordinates else None,
                                'timestamp': datetime.now().isoformat()
                            }

                            valid_elements.append(ContentElement(content, "image", clean_metadata))
                            self.logger.info(f"Processed image: {filename}, Page: {page_num}, Figure: {fig_num}")
                            break

            num_images = len(valid_elements)
            if num_images > 0:
                self.logger.info(f"Successfully extracted {num_images} images")
                # Debug log the extracted elements
                for elem in valid_elements:
                    self.logger.debug(f"Extracted: {elem.metadata.get('filename')}, "
                                      f"Page: {elem.metadata.get('page_number')}, "
                                      f"Figure: {elem.metadata.get('figure_number')}")
            else:
                self.logger.warning("No valid images were extracted!")
                self.logger.debug("Available files: " + ", ".join(f.name for f in image_files))

            return valid_elements

        except Exception as e:
            self.logger.error(f"Image extraction error: {str(e)}")
            raise

    def _extract_table_content(self):
        """Extract table content from PDF."""
        try:
            self.logger.info("Extracting table content...")
            table_elements = partition_pdf(
                filename=self.pdf_path,
                include_metadata=True,
                strategy="hi_res",
                infer_table_structure=True,
                include_table_data=True
            )

            valid_elements = []
            for element in table_elements:
                element_dict = element.to_dict() if hasattr(element, 'to_dict') else {}
                element_type = element_dict.get('type', '')
                category = element_dict.get('category', '')

                if category == "Table" or "Table" in str(element_type):
                    metadata = element_dict.get('metadata', {})
                    content = element_dict.get('text', '')

                    # Create clean metadata
                    clean_metadata = {
                        'page_number': metadata.get('page_number'),
                        'element_id': metadata.get('element_id', ''),
                        'content_type': 'table',
                        'coordinates': json.dumps(metadata.get('coordinates', {})),
                        'table_structure': json.dumps(metadata.get('table_structure', {})) if metadata.get(
                            'table_structure') else None,
                        'timestamp': datetime.now().isoformat()
                    }

                    valid_elements.append(ContentElement(content, "table", clean_metadata))
                    self.logger.info(f"Found table on page {clean_metadata['page_number']}")

            num_tables = len(valid_elements)
            if num_tables > 0:
                self.logger.info(f"Successfully extracted {num_tables} tables")
                for elem in valid_elements:
                    self.logger.debug(f"Extracted table: Page {elem.metadata.get('page_number')}")
            else:
                self.logger.warning("No valid tables were extracted!")

            return valid_elements

        except Exception as e:
            self.logger.error(f"Table extraction error: {str(e)}")
            raise

    # def _process_text(self, elements: List[ContentElement]):
    #     """Process text elements."""
    #     if not elements:
    #         return
    #
    #     documents = []
    #     for element in elements:
    #         # print(element.metadata)
    #         # print(element.content)
    #         # assert False
    #         chunks = self.text_splitter.split_text(element.content)
    #         for i, chunk in enumerate(chunks):
    #             analysis = self._process_with_retry(
    #                 self.text_agent,
    #                 f"Analyze this text:\n{chunk}"
    #             )
    #             metadata = {
    #                 **element.metadata,
    #                 'chunk_number': i + 1,
    #                 'total_chunks': len(chunks)
    #             }
    #             documents.append(Document(
    #                 page_content=analysis,
    #                 metadata=metadata
    #             ))
    #
    #     if documents:
    #         self.text_store = Chroma.from_documents(
    #             documents=documents,
    #             embedding=self.embeddings,
    #             persist_directory=str(self.kb_dir / "text_kb")
    #         )

    def _process_text(self, elements: List[ContentElement]):
        """Process text elements with direct chunking and embedding."""
        if not elements:
            return

        documents = []
        for element in elements:
            chunks = self.text_splitter.split_text(element.content)

            for i, chunk in enumerate(chunks):
                # Clean the metadata before creating Document
                metadata = self._clean_metadata({
                    **element.metadata,
                    'chunk_number': i + 1,
                    'total_chunks': len(chunks)
                })

                # Create document with cleaned metadata
                documents.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))

        if documents:
            self.text_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.kb_dir / "text_kb")
            )

    def _process_images(self, elements: List[ContentElement]):
        """Process image elements."""
        if not elements:
            return

        documents = []
        for element in elements:
            image_path = element.metadata.get('image_path')
            if image_path and os.path.exists(image_path):
                try:
                    # Get image analysis
                    analysis = self._process_with_retry(
                        self.vision_agent,
                        "Analyze this image in detail:",
                        images=[image_path]
                    )

                    # Clean metadata and remove complex structures
                    cleaned_metadata = self._clean_metadata({
                        'image_path': image_path,
                        'page_number': element.metadata.get('page_number'),
                        'content_type': 'image',
                        'timestamp': datetime.now().isoformat()
                    })

                    # Create document with cleaned metadata
                    documents.append(Document(
                        page_content=analysis,
                        metadata=cleaned_metadata
                    ))
                except Exception as e:
                    self.logger.error(f"Error processing image {image_path}: {str(e)}")
                    continue

        if documents:
            self.image_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.kb_dir / "image_kb")
            )

    def _process_tables(self, elements: List[ContentElement]):
        """Process table elements with agent analysis."""
        if not elements:
            return

        documents = []
        for element in elements:
            try:
                # Get table analysis from agent
                analysis = self._process_with_retry(
                    self.text_agent,
                    f"""Analyze this table in detail:
                    Table Content:
                    {element.content}

                    Please provide:
                    1. Structure and organization of the table
                    2. Key data points and relationships
                    3. Main findings or patterns
                    4. Significance in the document context
                    """
                )

                # Clean metadata and create document
                cleaned_metadata = self._clean_metadata({
                    **element.metadata,
                    'table_content': element.content  # Store original table content
                })

                documents.append(Document(
                    page_content=analysis,
                    metadata=cleaned_metadata
                ))

            except Exception as e:
                self.logger.error(f"Error processing table on page {element.metadata.get('page_number')}: {str(e)}")
                continue

        if documents:
            self.table_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.kb_dir / "table_kb")
            )

    def process_document(self):
        """Process the document with text, images, and tables."""
        try:
            self.logger.info(f"Processing document: {self.pdf_path}")

            # Process tables first
            table_elements = self._extract_table_content()
            if table_elements:
                self._process_tables(table_elements)

            # Extract and process images (existing code)
            image_elements = self._extract_image_content()
            if image_elements:
                self._process_images(image_elements)

            # Extract and process text (existing code)
            text_elements = self._extract_text_content()
            if text_elements:
                self._process_text(text_elements)

            self.logger.info("Document processing complete")

        except Exception as e:
            self.logger.error(f"Document processing error: {str(e)}")
            raise

    # Update the query method to include tables
    def query(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Query the knowledge base including tables."""
        try:
            result = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "sources": {
                    "text": {"content": [], "metadata": []},
                    "images": {"content": [], "metadata": []},
                    "tables": {"content": [], "metadata": []}  # Add tables
                }
            }

            # Query stores (add tables)
            for store_type, store in [
                ("text", self.text_store),
                ("images", self.image_store),
                ("tables", self.table_store)
            ]:
                if store:
                    collection = store._collection
                    doc_count = collection.count()
                    current_k = min(k, doc_count)
                    if current_k > 0:
                        docs = store.similarity_search(query, k=current_k)
                        for doc in docs:
                            result["sources"][store_type]["content"].append(doc.page_content)
                            result["sources"][store_type]["metadata"].append(doc.metadata)

            if not any(result["sources"][key]["content"] for key in result["sources"]):
                return {"error": "No relevant content found", "query": query}

            # Generate integrated response
            prompt = f"""Answer this query using the provided information:

    Query: {query}

    Text Information:
    {' '.join(result['sources']['text']['content'])}

    Image Information:
    {' '.join(result['sources']['images']['content'])}

    Table Information:
    {' '.join(result['sources']['tables']['content'])}

    Provide a comprehensive answer that combines all relevant information. And no need to tell from which information you are using from table, text and image"""

            result["response"] = self._process_with_retry(self.integration_agent, prompt)

            # Add source details to the response
            sources_info = "\n\nSource Information:"

            # Add text sources (existing code)
            if result["sources"]["text"]["content"]:
                sources_info += "\n\nText Chunks:"
                for i, (content, metadata) in enumerate(zip(
                        result["sources"]["text"]["content"],
                        result["sources"]["text"]["metadata"]
                ), 1):
                    sources_info += f"\n[Text {i}]"
                    sources_info += f"\n- Page: {metadata.get('page_number', 'unknown')}"
                    sources_info += f"\n- Chunk: {metadata.get('chunk_number', 'unknown')}/{metadata.get('total_chunks', 'unknown')}"

            # Add image sources (existing code)
            if result["sources"]["images"]["content"]:
                sources_info += "\n\nImages:"
                for i, (content, metadata) in enumerate(zip(
                        result["sources"]["images"]["content"],
                        result["sources"]["images"]["metadata"]
                ), 1):
                    sources_info += f"\n[Image {i}]"
                    sources_info += f"\n- Page: {metadata.get('page_number', 'unknown')}"
                    image_path = metadata.get('image_path', '')
                    if image_path:
                        sources_info += f"\n- File: {os.path.basename(image_path)}"

            # Add table sources
            if result["sources"]["tables"]["content"]:
                sources_info += "\n\nTables:"
                for i, (content, metadata) in enumerate(zip(
                        result["sources"]["tables"]["content"],
                        result["sources"]["tables"]["metadata"]
                ), 1):
                    sources_info += f"\n[Table {i}]"
                    sources_info += f"\n- Page: {metadata.get('page_number', 'unknown')}"

            # Combine response with source information
            result["response"] = result["response"] + sources_info
            return result

        except Exception as e:
            self.logger.error(f"Query error: {str(e)}")
            return {"error": str(e), "query": query}

def main():
    pdf_path = "/home/aifahim/PycharmProjects/test-agentic-rag/src/data/pdfs/Otani_Toward_Verifiable_and_Reproducible_Human_Evaluation_for_Text-to-Image_Generation_CVPR_2023_paper.pdf"
    output_dir = "rag_output"

    try:
        # Initialize and process document
        rag = MultimodalRAG(pdf_path, output_dir)
        rag.process_document()

        # Example query
        query = "What are the main findings in the document?"
        result = rag.query(query)

        # Print results with formatting
        if "error" in result:
            print("\nError:", result["error"])
        else:
            print("\nQuery:", result["query"])
            print("\nResponse:", result["response"])

    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()