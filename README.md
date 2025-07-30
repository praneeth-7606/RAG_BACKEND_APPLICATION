# InsureRAG - FastAPI Backend

This is the backend service for the Insurance Policy Q&A application. It is a robust FastAPI server that powers a complete Retrieval-Augmented Generation (RAG) pipeline, handling document ingestion, validation, processing, and AI-powered chat.

### Live Demo & Repository

- **Live Backend URL**: `https://rag-backend-application-2.onrender.com`
- **Frontend Repository**: `[Link to your Frontend GitHub Repo]`

---

### Core Features

- **Robust API Endpoints**: Provides endpoints for document upload (`/upload`), health checks (`/health`), and AI-powered chat (`/chat`).
- **AI-Powered Validation**: Before processing, an LLM call validates that uploaded documents are genuinely related to insurance, rejecting irrelevant files.
- **RAG Pipeline**: Implements a complete RAG workflow using LangChain:
    1.  **Text Extraction**: Reliably extracts text from both PDF and TXT files.
    2.  **Chunking**: Logically splits documents into smaller chunks with overlap to retain context.
    3.  **Embedding**: Generates vector embeddings for each chunk using Google's models.
    4.  **Vector Storage**: Stores and indexes vectors in a persistent, cloud-based Pinecone database.
    5.  **Retrieval & Generation**: Retrieves relevant document chunks based on user queries and uses a Gemini LLM to generate accurate, context-aware answers.

---

### Tech Stack

- **Framework**: FastAPI
- **Language**: Python 3.10+
- **RAG & AI**: LangChain, Langchain-Google-Genai
- **LLM**: Google Gemini-Pro-Flash
- **Embedding Model**: Google `models/embedding-001`
- **Vector Database**: Pinecone
- **PDF Parsing**: PyMuPDF
- **Deployment**: Render

---

### Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone [Your-Repo-URL]
    cd [Your-Repo-Name]/server
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**:
    Create a `.env` file in the `server` directory by copying the example:
    ```bash
    cp .env.example .env
    ```
    Fill in the required API keys in the `.env` file:
    ```env
    GOOGLE_API_KEY="your_google_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    PINECONE_ENVIRONMENT="environment name" 
    PINECONE_INDEX_NAME="index name"
    ```

### Running the Application Locally

To run the FastAPI server for local development:
```bash
uvicorn main:app --reload
