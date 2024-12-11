from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.chroma import ChromaDb


class LLMRAgentService:
    def __init__(self, pdf_path: str, collection: str, db_path: str, enable_rag=True, persistent=True):
        self.knowledge_base = PDFKnowledgeBase(
            path=pdf_path,
            vector_db=ChromaDb(
                collection=collection,
                path=db_path,
                persistent_client=persistent,
            ),
        )
        self.knowledge_base.load(recreate=False)

        self.agent = Agent(
            model=OpenAIChat(model="gpt-3.5-turbo"),
            description="You are a helpful assistant with access to a knowledge base.",
            instructions=[
                "Use the provided knowledge base to answer questions when relevant.",
                "If you cannot find an answer in the knowledge base, respond with 'I don't know.'",
                "Format responses in markdown."
            ],
            markdown=True,
            enable_rag=enable_rag,
            knowledge=self.knowledge_base,
        )
