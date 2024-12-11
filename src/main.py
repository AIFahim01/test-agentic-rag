from pathlib import Path
from config import PDFS, SIMILARITY_TOP_K, OPENAI_API_KEY
from llm.llm_init import initialize_llm
from indexing.chroma_manager import ChromaManager
from indexing.retriever_builder import build_retriever
from agents.agent_factory import create_agent
from services.embeddings_services import get_embeddings
from services.text_pre_processing import extract_text_from_pdf, chunk_text

if __name__ == "__main__":
    llm = initialize_llm()

    all_chunks = []
    for pdf_path in PDFS:
        print(f"Processing paper: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        doc_id = Path(pdf_path).stem
        chunks = chunk_text(text)

        # Get embeddings for these chunks
        embeddings = get_embeddings(chunks)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding
            })


    chroma_manager = ChromaManager(collection_name="papers_collection", persist_directory=".chromadb")
    chroma_manager.add_documents(all_chunks)


    obj_retriever = build_retriever(chroma_manager, similarity_top_k=SIMILARITY_TOP_K)


    with open("src/agents/prompts/system_prompt.txt", "r") as f:
        system_prompt = f.read()


    agent = create_agent(tool_retriever=obj_retriever, llm=llm, system_prompt=system_prompt, verbose=True)


    query1 = "Tell me about the evaluation dataset used in LongLoRA, and then tell me about the evaluation results."
    response1 = agent.query(query1)
    print("Response 1:", response1)

