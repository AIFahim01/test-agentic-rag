from chromadb.config import Settings
from chromadb import Client

client = Client(Settings(
    persist_directory="data/chromadb",  # Use an explicit persist directory
    chroma_db_impl="duckdb+parquet"    # Define the database implementation
))
print("ChromaDB Client initialized successfully.")
