from indexing.chroma_manager import ChromaManager
from utils.embeddings import get_embeddings

def build_retriever(chroma_manager: ChromaManager, similarity_top_k=3):
    def retriever(query: str):
        query_embedding = get_embeddings(query)
        results = chroma_manager.query(query_embedding, top_k=similarity_top_k)
        return results
    return retriever
