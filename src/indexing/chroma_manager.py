import chromadb
from chromadb.config import Settings


class ChromaManager:
    def __init__(self, collection_name="papers_collection", persist_directory=".chromadb"):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        self.collection = self._get_or_create_collection(collection_name)

    def _get_or_create_collection(self, name):
        if name not in [c.name for c in self.client.list_collections()]:
            return self.client.create_collection(name=name)
        return self.client.get_collection(name=name)

    def add_documents(self, docs):
        # docs is a list of dict with keys: doc_id, chunk_id, text, embedding
        for doc in docs:
            self.collection.add(
                embeddings=[doc["embedding"]],
                documents=[doc["text"]],
                ids=[f"{doc['doc_id']}-{doc['chunk_id']}"],
                metadatas=[{"doc_id": doc["doc_id"], "chunk_id": doc["chunk_id"]}]
            )

    def query(self, query_embedding, top_k=3):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
