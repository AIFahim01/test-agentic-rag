from chromadb.client import Client
from flask import current_app
from typing import List, Dict
import tiktoken
import numpy as np
import openai



class EmbeddingService:
    def __init__(self):
        """
        Initialize the EmbeddingService with ChromaDB client and tokenizer.
        """
        self.client = Client()  # Initialize the ChromaDB client
        self.collection_name = "document_embeddings"
        self.collection = self.client.get_or_create_collection(self.collection_name)
        self.encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
        self.max_tokens = 512  # Maximum tokens per chunk

    def create_chunks(self, text: str) -> List[str]:
        """
        Split text into manageable chunks based on token limit.
        Args:
            text (str): Input document text.
        Returns:
            List[str]: List of text chunks.
        """
        chunks = []
        current_chunk = []
        current_length = 0

        # Split text into sentences
        sentences = text.split('. ')

        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))

            if current_length + sentence_tokens > self.max_tokens:
                # Save current chunk and start new one
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text chunk using OpenAI API.
        Args:
            text (str): Input text to generate embedding for.
        Returns:
            List[float]: The embedding vector.
        """
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']

    def process_document(self, document_id: int, text: str):
        """
        Process document text into chunks and store embeddings in ChromaDB.
        Args:
            document_id (int): The ID of the document being processed.
            text (str): The full text of the document.
        """
        chunks = self.create_chunks(text)

        for idx, chunk in enumerate(chunks):
            embedding = self.generate_embedding(chunk)
            # Add chunk to ChromaDB collection
            self.collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"document_id": document_id, "chunk_id": idx}]
            )

    def find_relevant_chunks(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Find the most relevant document chunks for a query using ChromaDB.
        Args:
            query (str): The query text.
            limit (int): The number of top results to return.
        Returns:
            List[Dict]: List of relevant chunks with similarity scores.
        """
        query_embedding = self.generate_embedding(query)

        # Perform similarity search using ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )

        return [
            {
                "chunk_text": result["document"],
                "document_id": result["metadata"]["document_id"],
                "similarity": result["score"]
            }
            for result in results["matches"]
        ]
