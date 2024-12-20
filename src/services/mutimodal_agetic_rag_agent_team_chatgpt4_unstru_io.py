import os
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import openai
from dotenv import load_dotenv
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.vectorstores.utils import filter_complex_metadata

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_to_json

from phi.agent import Agent
from phi.knowledge.langchain import LangChainKnowledgeBase
from phi.model.openai import OpenAIChat

from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def setup_logging(output_dir: str):
    """Setup logging configuration."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(exist_ok=True)

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    file_handler = logging.FileHandler(
        log_dir / "processing.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger


class ExtractedContent:
    """Class to hold extracted content with detailed metadata."""
    def __init__(self, content, element_type, category, metadata, image_path=None):
        self.content = content
        self.element_type = element_type
        self.category = category
        self.metadata = metadata
        self.image_path = image_path
        # Make compatible with chunk_by_title
        self.text = content

    @classmethod
    def from_unstructured_element(cls, element, image_dir: str = None):
        """Create from an unstructured.io element with full metadata."""
        if isinstance(element, str):
            return cls(
                content=element,
                element_type="text",
                category="Text",
                metadata={
                    'element_type': 'text',
                    'category': 'Text',
                    'text_length': len(element),
                    'processing_timestamp': datetime.now().isoformat()
                }
            )

        element_dict = element.to_dict()
        content = element_dict.get('text', '')
        element_type = element_dict.get('type', '')
        category = element_dict.get('category', '')

        metadata = {
            'element_id': element_dict.get('metadata', {}).get('element_id'),
            'page_number': element_dict.get('metadata', {}).get('page_number'),
            'filename': element_dict.get('metadata', {}).get('filename'),
            'element_type': element_type,
            'category': category,
            'text_length': len(content),
            'processing_timestamp': datetime.now().isoformat()
        }

        # Handle image path
        image_path = None
        if category == "Image" and image_dir:
            page_num = metadata.get('page_number')
            if page_num:
                possible_images = list(Path(image_dir).glob(f"*page{page_num}*.jpg"))
                if possible_images:
                    image_path = str(possible_images[0])
                    metadata['image_path'] = image_path

        return cls(
            content=content,
            element_type=element_type,
            category=category,
            metadata=metadata,
            image_path=image_path
        )


class MultimodalRAG:
    def __init__(self, pdf_path: str, output_dir: str):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "extracted_images"
        self.kb_dir = self.output_dir / "knowledge_base"

        # Setup logging
        self.logger = setup_logging(str(self.output_dir))
        self.logger.info(f"Initializing MultimodalRAG for PDF: {self.pdf_path}")

        # Create directories
        self._setup_directories()

        # Initialize components
        self._initialize_components()

        # Initialize agents
        self._initialize_agents()

    def _setup_directories(self):
        """Setup required directories."""
        directories = [
            self.output_dir,
            self.image_dir,
            self.kb_dir
        ]
        for directory in directories:
            directory.mkdir(exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")

    def _initialize_components(self):
        """Initialize embeddings and text splitter."""
        self.logger.info("Initializing components...")
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )

        self.text_vectorstore = None
        self.image_vectorstore = None
        self.table_vectorstore = None

    def _initialize_agents(self):
        """Initialize specialized agents."""
        self.logger.info("Initializing agent team...")

        self.text_agent = Agent(
            name="Text Analyzer",
            role="Process and understand textual content",
            model=OpenAIChat(id="gpt-4"),
            instructions=[
                "Analyze and extract key information from text.",
                "Focus on main ideas, arguments, and findings.",
                "Identify important relationships and connections.",
            ],
            markdown=True
        )

        self.vision_agent = Agent(
            name="Visual Content Analyzer",
            role="Analyze images and visual content",
            model=OpenAIChat(id="gpt-4o"),
            instructions=[
                "Analyze images, figures, and tables in detail.",
                "Extract quantitative and qualitative information.",
                "Describe visual elements and their relationships.",
                "For tables: analyze structure, headers, and data relationships.",
                "For figures: describe visual patterns and key findings.",
            ],
            markdown=True
        )

        self.integration_agent = Agent(
            name="Integration Specialist",
            role="Combine insights across modalities",
            model=OpenAIChat(id="gpt-4"),
            team=[self.text_agent, self.vision_agent],
            instructions=[
                "Combine insights from text and visual content.",
                "Create comprehensive understanding across modalities.",
                "Include source locations and metadata in responses.",
                "When tables are relevant, include their numerical data.",
                "When images are relevant, reference their visual content.",
            ],
            markdown=True
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _process_with_retry(self, agent: Agent, prompt: str, images: List[str] = None) -> str:
        """Process content with retry logic."""
        try:
            if images:
                response = agent.run(prompt, images=images)
            else:
                response = agent.run(prompt)
            return response.content
        except Exception as e:
            self.logger.error(f"Error in agent processing: {str(e)}")
            raise

    # def process_document(self):
    #     """Process document with proper image and table extraction."""
    #     try:
    #         self.logger.info("Starting document processing...")
    #
    #         # Extract elements
    #         self.logger.info("Extracting document elements...")
    #         elements = partition_pdf(
    #             filename=self.pdf_path,
    #             include_metadata=True,
    #             strategy="hi_res",
    #             hi_res_model_name="yolox",
    #             infer_table_structure=True,
    #             extract_images_in_pdf=True,
    #             image_output_dir_path=str(self.image_dir),
    #             extract_image_block_types=["Image", "Table"]
    #         )
    #
    #         text_elements = []
    #         image_elements = []
    #         table_elements = []
    #
    #         # Sort elements into categories
    #         for element in elements:
    #             extracted = ExtractedContent.from_unstructured_element(
    #                 element,
    #                 image_dir=str(self.image_dir)
    #             )
    #
    #             if extracted.category == "Image":
    #                 image_elements.append(extracted)
    #             elif extracted.category == "Table" or "Table" in extracted.element_type:
    #                 table_elements.append(extracted)
    #             elif extracted.category not in ["Header", "Footer"]:
    #                 text_elements.append(extracted)
    #
    #         # Process extracted contents
    #         if text_elements:
    #             self.logger.info(f"Processing {len(text_elements)} text elements...")
    #             self._process_text_contents(text_elements)
    #
    #         if image_elements:
    #             self.logger.info(f"Processing {len(image_elements)} image elements...")
    #             self._process_image_contents(image_elements)
    #
    #         if table_elements:
    #             self.logger.info(f"Processing {len(table_elements)} table elements...")
    #             self._process_table_contents(table_elements)
    #
    #         self.logger.info("Document processing complete")
    #
    #     except Exception as e:
    #         self.logger.error(f"Error processing document: {str(e)}")
    #         raise

    def process_document(self):
        """Process document with corrected image extraction."""
        try:
            self.logger.info("Starting document processing...")

            # Ensure image directory exists
            self.image_dir.mkdir(parents=True, exist_ok=True)

            # First pass: Extract media content
            self.logger.info("Extracting media content...")
            media_elements = partition_pdf(
                filename=self.pdf_path,
                include_metadata=True,
                strategy="hi_res",
                hi_res_model_name="yolox",
                infer_table_structure=True,
                extract_images_in_pdf=True,
                extract_image_block_output_dir=str(self.image_dir),  # Changed parameter name
                extract_image_block_types=["Image", "Table", "Figure"],
                include_image_data=True,
                include_chart_data=True,  # Added for better image extraction
                include_table_data=True,  # Added for better table extraction
                include_page_breaks=True,
                languages=["eng"]
            )

            # Process media elements
            image_elements = []
            table_elements = []

            for element in media_elements:
                # Convert element to dict for consistent attribute access
                element_dict = element.to_dict() if hasattr(element, 'to_dict') else {}
                element_type = element_dict.get('type', '')
                category = element_dict.get('category', '')

                # Get page number from metadata
                metadata_dict = element_dict.get('metadata', {})
                page_num = metadata_dict.get('page_number')

                if category == "Image" or "Figure" in element_type:
                    self.logger.info(f"Found image on page {page_num}")

                    # Check for actual image file with multiple patterns
                    if page_num:
                        # Try multiple patterns and formats
                        patterns = [
                            (f"*_page_{page_num}*.*", "exact page match"),
                            (f"*_p{page_num}*.*", "p prefix"),
                            (f"*_page{page_num}*.*", "page prefix"),
                            (f"*_{page_num}*.*", "number only"),
                            (f"*{page_num}*.*", "any match")
                        ]

                        image_path = None
                        for pattern, desc in patterns:
                            matches = list(self.image_dir.glob(pattern))
                            if matches:
                                image_path = str(matches[0])
                                self.logger.info(f"Found image using {desc}: {image_path}")
                                break

                        if image_path:
                            extracted = ExtractedContent.from_unstructured_element(
                                element,
                                image_dir=str(self.image_dir)
                            )
                            extracted.image_path = image_path
                            image_elements.append(extracted)
                            self.logger.info(f"Successfully linked image file for page {page_num}")
                        else:
                            # List all files in image directory for debugging
                            existing_files = list(self.image_dir.glob("*"))
                            self.logger.warning(
                                f"No image file found for page {page_num}. Directory contains: {[f.name for f in existing_files]}")

                elif category == "Table" or "Table" in element_type:
                    self.logger.info(f"Found table on page {page_num}")
                    table_elements.append(ExtractedContent.from_unstructured_element(element))

            # Second pass: Extract and chunk text
            self.logger.info("Processing text content...")
            text_elements = partition_pdf(
                filename=self.pdf_path,
                include_metadata=True,
                strategy="hi_res",
                hi_res_model_name="yolox",
                chunking_strategy="by_title",
                max_characters=3000,
                combine_text_under_n_chars=200,
                multipage_sections=True,
                extract_images_in_pdf=False
            )

            # Process text elements
            valid_text_elements = []
            for element in text_elements:
                element_dict = element.to_dict() if hasattr(element, 'to_dict') else {}
                category = element_dict.get('category', '')

                if category not in ["Header", "Footer", "Image", "Table"]:
                    extracted = ExtractedContent.from_unstructured_element(element)
                    valid_text_elements.append(extracted)

            # Process each content type
            if valid_text_elements:
                self.logger.info(f"Processing {len(valid_text_elements)} text elements...")
                self._process_text_contents(valid_text_elements)

            if image_elements:
                self.logger.info(f"Found {len(image_elements)} valid images with files")
                self._process_image_contents(image_elements)
            else:
                self.logger.warning("No valid image files were found")

            if table_elements:
                self.logger.info(f"Processing {len(table_elements)} table elements...")
                self._process_table_contents(table_elements)

            self.logger.info("Document processing complete")

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def _clean_metadata(self, metadata: dict) -> dict:
        """Clean metadata by removing None values and complex types."""
        if not isinstance(metadata, dict):
            self.logger.warning(f"Received non-dict metadata of type {type(metadata)}")
            return {}
        filtered = {k: v for k, v in metadata.items() if v is not None}
        cleaned = {}
        for k, v in filtered.items():
            if isinstance(v, (str, int, float, bool)):
                cleaned[k] = v
            elif isinstance(v, (dict, list)):
                cleaned[k] = str(v)
        return cleaned

    def _process_text_contents(self, contents):
        """Process text contents by splitting and analyzing them."""
        if not contents:
            return

        # Limit for demonstration
        contents = contents[:2]
        self.logger.info(f"Processing {len(contents)} text elements...")
        text_documents = []

        for idx, content in enumerate(contents, 1):
            if not hasattr(content, 'content') or not hasattr(content, 'metadata'):
                self.logger.warning(f"Invalid content structure at index {idx}")
                continue

            chunks = self.text_splitter.split_text(content.content)
            base_metadata = self._clean_metadata(content.metadata)

            for chunk_idx, chunk in enumerate(chunks, 1):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_number': chunk_idx,
                    'total_chunks': len(chunks),
                    'chunk_length': len(chunk)
                })

                try:
                    analysis = self._process_with_retry(
                        self.text_agent,
                        f"Analyze this text:\n{chunk}"
                    )
                    doc = Document(
                        page_content=analysis,
                        metadata=chunk_metadata
                    )
                    text_documents.append(doc)

                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_idx} of content {idx}: {str(e)}")
                    continue

        if text_documents:
            try:
                self.text_vectorstore = Chroma.from_documents(
                    documents=text_documents,
                    embedding=self.embeddings,
                    persist_directory=str(self.kb_dir / "text_kb")
                )
                self.logger.info("Successfully created text vectorstore")
            except Exception as e:
                self.logger.error(f"Error creating text vectorstore: {str(e)}")
                raise
        else:
            self.logger.warning("No valid text documents were created")

    def _process_image_contents(self, contents):
        """Process image contents with improved file handling."""
        if not contents:
            return

        self.logger.info(f"Processing {len(contents)} image elements...")
        image_documents = []

        for idx, content in enumerate(contents, 1):
            try:
                # Get image path with fallback options
                image_path = None
                if hasattr(content, 'image_path') and content.image_path:
                    image_path = content.image_path
                elif hasattr(content, 'metadata'):
                    page_num = content.metadata.get('page_number')
                    if page_num:
                        # Try multiple file patterns and formats
                        patterns = [
                            f"*page_{page_num}*.jpg",
                            f"*page_{page_num}*.png",
                            f"*_{page_num}*.jpg",
                            f"*_{page_num}*.png"
                        ]
                        for pattern in patterns:
                            matches = list(self.image_dir.glob(pattern))
                            if matches:
                                image_path = str(matches[0])
                                break

                if not image_path or not os.path.exists(image_path):
                    self.logger.warning(f"No valid image file found for image {idx}")
                    continue

                metadata = content.metadata if isinstance(content.metadata, dict) else {}
                metadata['image_path'] = image_path
                metadata = self._clean_metadata(metadata)

                prompt = """Analyze this image in detail:
    1. What type of visual content is this (e.g., figure, chart, photo)?
    2. What are the main visual elements and their relationships?
    3. What key information or findings does this image convey?
    4. Are there any notable patterns or trends?
    5. How does this relate to the document's content?"""

                analysis = self._process_with_retry(
                    self.vision_agent,
                    prompt,
                    images=[image_path]
                )

                doc = Document(
                    page_content=analysis,
                    metadata=metadata
                )
                image_documents.append(doc)
                self.logger.info(f"Successfully processed image {idx}")

            except Exception as e:
                self.logger.error(f"Error processing image {idx}: {str(e)}")
                continue

        if image_documents:
            try:
                self.image_vectorstore = Chroma.from_documents(
                    documents=image_documents,
                    embedding=self.embeddings,
                    persist_directory=str(self.kb_dir / "image_kb")
                )
                self.logger.info("Successfully created image vectorstore")
            except Exception as e:
                self.logger.error(f"Error creating image vectorstore: {str(e)}")
                raise

    def _process_table_contents(self, contents):
        """Process table contents using text agent for analysis."""
        if not contents:
            return

        # Limit for demonstration
        contents = contents[:2]
        self.logger.info(f"Processing {len(contents)} table elements...")
        table_documents = []

        for idx, content in enumerate(contents, 1):
            try:
                metadata = self._clean_metadata(content.metadata)
                prompt = f"""Analyze this table content:
{content.content}

1. What is the structure and organization of this data?
2. What are the key findings or patterns?
3. How does this information relate to the document's context?"""

                analysis = self._process_with_retry(
                    self.text_agent,
                    prompt
                )

                doc = Document(
                    page_content=analysis,
                    metadata=metadata
                )
                table_documents.append(doc)
                self.logger.info(f"Successfully processed table {idx}")

            except Exception as e:
                self.logger.error(f"Error processing table {idx}: {str(e)}")
                continue

        if table_documents:
            try:
                self.table_vectorstore = Chroma.from_documents(
                    documents=table_documents,
                    embedding=self.embeddings,
                    persist_directory=str(self.kb_dir / "table_kb")
                )
                self.logger.info("Successfully created table vectorstore")
            except Exception as e:
                self.logger.error(f"Error creating table vectorstore: {str(e)}")
                raise

    def _create_metadata_aware_prompt(self, query: str, sources: Dict) -> str:
        """Create a prompt that includes metadata context."""
        prompt = f"""Answer this query using the provided information and metadata:

Query: {query}

Available Information:
"""
        for source_type, data in sources.items():
            if data["content"]:
                prompt += f"\n{source_type.upper()} CONTENT:\n"
                for content, metadata in zip(data["content"], data["metadata"]):
                    location_info = f"Page {metadata.get('page_number', 'unknown')}"
                    if metadata.get('coordinates'):
                        coords = metadata['coordinates']
                        location_info += f" (Region: {coords.get('system', 'unknown')})"
                    prompt += f"\nSource [{location_info}]:\n{content}\n"

        prompt += "\nPlease provide a comprehensive answer that includes relevant source locations and context."
        return prompt

    def query(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Query with detailed metadata return."""
        try:
            self.logger.info(f"Processing query: {query}")

            result = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "sources": {
                    "text": {"content": [], "metadata": []},
                    "images": {"content": [], "metadata": []},
                    "tables": {"content": [], "metadata": []}
                }
            }

            # Search text vectorstore
            if self.text_vectorstore:
                text_docs = self.text_vectorstore.similarity_search(query, k=k)
                for doc in text_docs:
                    result["sources"]["text"]["content"].append(doc.page_content)
                    result["sources"]["text"]["metadata"].append(doc.metadata)

            # Search image vectorstore
            if self.image_vectorstore:
                image_docs = self.image_vectorstore.similarity_search(query, k=k)
                for doc in image_docs:
                    result["sources"]["images"]["content"].append(doc.page_content)
                    result["sources"]["images"]["metadata"].append(doc.metadata)

            # Search table vectorstore
            if self.table_vectorstore:
                table_docs = self.table_vectorstore.similarity_search(query, k=k)
                for doc in table_docs:
                    result["sources"]["tables"]["content"].append(doc.page_content)
                    result["sources"]["tables"]["metadata"].append(doc.metadata)

            # Create metadata-aware prompt
            prompt = self._create_metadata_aware_prompt(query, result["sources"])

            # Get integrated response
            self.logger.info("Generating integrated response...")
            response = self._process_with_retry(self.integration_agent, prompt)
            result["response"] = response

            # Save query log
            self._save_query_log(result)

            # Add source summary
            source_summary = {
                "text_sources": len(result["sources"]["text"]["content"]),
                "image_sources": len(result["sources"]["images"]["content"]),
                "table_sources": len(result["sources"]["tables"]["content"])
            }
            result["source_summary"] = source_summary

            self.logger.info("Query processing complete")
            return result

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.logger.error(error_msg)
            return {
                "error": error_msg,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }

    def _save_query_log(self, result: Dict[str, Any]) -> None:
        """Save query results and metadata to a JSON log file."""
        query_log_path = self.output_dir / "query_log.json"

        try:
            # Load existing logs or create a new list
            if query_log_path.exists():
                with open(query_log_path, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []

            logs.append(result)

            with open(query_log_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)

            self.logger.debug("Query log saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving query log: {str(e)}")

def main():
    pdf_path = "/home/aifahim/PycharmProjects/test-agentic-rag/src/data/pdfs/Otani_Toward_Verifiable_and_Reproducible_Human_Evaluation_for_Text-to-Image_Generation_CVPR_2023_paper.pdf"
    output_dir = "./multimodal_rag_output"

    try:
        rag = MultimodalRAG(pdf_path, output_dir)

        print("\nProcessing document...")
        rag.process_document()

        query = input("\nEnter your query (or press Enter for default): ").strip()
        if not query:
            query = "Comparision between Qualification Annotator performance Stable Diffusion Real image"

        print(f"\nProcessing query: {query}")
        result = rag.query(query)

        if "error" in result:
            print("\nError:", result["error"])
        else:
            print("\nResponse:", result["response"])
            print("\nSource Summary:")
            print(json.dumps(result["source_summary"], indent=2))

            print("\nDetailed Sources:")
            for source_type, data in result["sources"].items():
                if data["metadata"]:
                    print(f"\n{source_type.upper()} Sources:")
                    for metadata in data["metadata"]:
                        page = metadata.get("page_number", "unknown")
                        coords = metadata.get("coordinates", {})
                        print(f"- Page {page}")
                        if coords and coords.get("points"):
                            print(f"  Location: {coords['points']}")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
