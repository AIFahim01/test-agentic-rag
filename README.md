# Test Agentic RAG

## Project Overview

This project demonstrates the implementation of Retrieval-Augmented Generation (RAG) using both Agentic and Vanilla approaches to retrieve context from a PDF knowledge base and answer user queries using OpenAI's GPT models. It integrates ChromaDB for vector storage and provides tools for indexing, querying, and managing a knowledge base.

### Supported RAG Modes
1. **Agentic RAG**: Uses `phi.agent` for seamless integration with a knowledge base and additional tools.
2. **Vanilla RAG**: Implements a simpler RAG workflow with direct query processing using LangChain's `ChatOpenAI`.

The knowledge base is built from PDF files, and embeddings are stored in ChromaDB for similarity-based searches.

## Features

* **PDF Indexing**: Automatically extract and store embeddings from PDFs
* **Agentic RAG**: Utilize advanced tools like `phi.agent` for enhanced query handling
* **Vanilla RAG**: Direct interaction with LangChain's `ChatOpenAI`
* **Interactive Session**: Query the knowledge base in real-time
* **Text Cleaning Tool**: Preprocess text for embedding using a custom cleaning pipeline

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/test-agentic-rag.git
cd test-agentic-rag
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root with the following contents:

```env
OPENAI_API_KEY=<openai_api_key>
```

## Usage

### 1. Start the Application

```bash
python src/main.py
```

### 2. Interactive Session

* Type queries in the interactive session
* Type `exit` or `bye` to quit

### 3. RAG Modes

* **Vanilla RAG**: Uses LangChain's `ChatOpenAI`
* **Agentic RAG**: Uses `phi.agent` (can be enabled in the main script)

## Project Structure

```
test-agentic-rag/
├── src/
│   ├── main.py               # Main entry point
│   ├── services/
│   │   ├── agentic_rag_service.py    # Agentic RAG implementation
│   │   ├── vanila_rag_service.py     # Vanilla RAG implementation
│   │   ├── embeddings_services.py    # Handles embeddings and vector storage
│   │   ├── text_cleaner_service.py   # Preprocesses text for embeddings
│   │   └── knowledge_base_service.py # Pdf knowledgebase for Agentic RAG
│   └── data/
│       └── pdfs/             # Store PDF files here
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
└── README.md                  # This README file
```

## Configuration

### Environment Variables
* Set the required variables in the `.env` file

### PDF Directory
* Add PDF files to `src/data/pdfs`

### ChromaDB
* Ensure ChromaDB is properly configured in `DB_PATH`

## How It Works

### 1. Indexing PDFs
* PDFs are processed, and their text is chunked
* Text chunks are cleaned and converted to embeddings
* Embeddings are stored in ChromaDB for similarity search

### 2. Querying
* User queries are matched against the stored embeddings
* Context from the most relevant documents is retrieved
* A final response is generated using OpenAI's GPT model

### 3. RAG Modes
* **Agentic RAG**: Advanced integration with tools and agent-based workflows
* **Vanilla RAG**: Simple, lightweight implementation using LangChain
