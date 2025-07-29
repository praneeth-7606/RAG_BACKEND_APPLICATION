# import os
# from pinecone import Pinecone, ServerlessSpec
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_core.documents import Document
# from langchain_text_splitters  import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv
# load_dotenv()
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# pc = Pinecone()
# index_name = "insurance-rag-index"
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=768,  
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
# pinecone_index = pc.Index(index_name)


# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
# )

# vectorstore = PineconeVectorStore(
#     index=pinecone_index,
#     embedding=embeddings,
#     namespace="pdf-docs" 
# )

# def upload_pdf_to_pinecone(pdf_text: str,source_url: str,pdf_name: str,namespace: str = "files",chunk_size: int = 1000,chunk_overlap: int = 200):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         is_separator_regex=False,
#     )

#     chunks = text_splitter.split_text(pdf_text)

#     documents = []
#     for i, chunk in enumerate(chunks):
#         documents.append(
#             Document(
#                 page_content=chunk,
#                 metadata={
#                     "source_url": source_url,
#                     "pdf_name": pdf_name,
#                     "chunk_id": i  
#                 }
#             )
#         )


#     vectorstore.add_documents(documents=documents, namespace=namespace)
#     print(f"Successfully uploaded {len(documents)} chunks to the '{namespace}' namespace.")

# def answer_with_context(
#     question: str,
#     vectorstore: PineconeVectorStore,
#     namespace: str = "files",
#     k: int = 4
# ):

#     retrieved_docs = vectorstore.similarity_search(
#         query=question, k=k, namespace=namespace
#     )
#     context = ""
#     for doc in retrieved_docs:
#         context += f"Source: {doc.metadata.get('source_url', 'N/A')}\n"
#         context += f"PDF: {doc.metadata.get('pdf_name', 'N/A')}\n"
#         context += f"Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}\n"
#         context += f"Content: {doc.page_content}\n---\n"

#     # Create the prompt
#     prompt = f"""
#     Answer the following question based on the provided context.
#     Include the source, PDF name, and chunk ID in your answer.
#     Context:
#     {context}
#     Question: {question}
#     Answer:
#     """

#     llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)


#     response = llm.invoke(prompt)

#     return response.content
# # --- Example Usage ---
# if __name__ == '__main__':
#     # Example data (longer text to demonstrate chunking)
#     sample_pdf_text = """
#     This is a long sample text to demonstrate the chunking functionality.
#     By breaking this document into smaller, more manageable pieces, we can improve
#     the performance of our retrieval system. Each chunk will be embedded separately,
#     allowing for more precise context matching when a user submits a query.
#     The overlap between chunks helps to ensure that no important information is lost
#     at the boundaries. This is a common practice in building robust RAG systems.
#     This process is repeated for every document you wish to add to the vector store.
#     The final step is to upload these chunks to the specified Pinecone namespace.
#     """
#     sample_source_url = "https://example.com/long_sample.pdf"
#     sample_pdf_name = "long_sample.pdf"

#     # Upload the data with chunking
#     print(f"Uploading and chunking document '{sample_pdf_name}'...")
#     upload_pdf_to_pinecone(
#         pdf_text=sample_pdf_text,
#         source_url=sample_source_url,
#         pdf_name=sample_pdf_name,
#         namespace="files",  # Explicitly defining the namespace for clarity
#         chunk_size=150,
#         chunk_overlap=30
#     )
#     print("Upload complete.")

#     # You can verify the upload by performing a similarity search
#     print("\n--- Verification Search ---")
#     retrieved_docs = vectorstore.similarity_search(
#         "What is a common practice in RAG systems?",
#         k=30,
#         namespace="files"
#     )

#     print("\nVerification search results:")
#     for doc in retrieved_docs:
#         print(f"- Content: {doc.page_content}")
#         print(f"  Metadata: {doc.metadata}\n")

#     # Answer a question using the new function
#     print("\n--- Answering with Context ---")
#     question = "What is a common practice in building robust RAG systems and why is it important?"
#     answer = answer_with_context(question, vectorstore, namespace="files", k=40)
#     print(f"Question: {question}")
#     print(f"Answer: {answer}")






import os
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uvicorn

# --- 1. INITIALIZATION ---

# Load environment variables from .env file
load_dotenv()

# Configure FastAPI app
app = FastAPI(
    title="RAG API with FastAPI",
    description="An API for uploading documents and chatting with a RAG model.",
    version="1.0.0",
)




origins = [
    "http://localhost:3000",  # React/Next.js default development port
    # Add your production frontend URL here later if you deploy
    # "https://your-production-site.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "insurance-rag-index"

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=768,  # Dimension for "models/embedding-001"
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
pinecone_index = pc.Index(index_name)
print("Pinecone index is ready.")

# Initialize Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Initialize Pinecone VectorStore
vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings,
    namespace="files"  # Default namespace for the vectorstore instance
)

# Initialize the Language Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# --- 2. Pydantic Models for Request/Response ---

# MODIFIED: ChatRequest now only requires the question.
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    source_documents: list[dict]

class UploadResponse(BaseModel):
    filename: str
    message: str
    chunks_uploaded: int


# --- 3. CORE RAG LOGIC (Helper Functions) ---

def upload_data_to_pinecone(
    text_data: str,
    source_filename: str,
    namespace: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """Chunks text data and uploads it to Pinecone."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(text_data)

    documents = []
    for i, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source_filename": source_filename,
                    "chunk_id": i
                }
            )
        )

    vectorstore.add_documents(documents=documents, namespace=namespace)
    return len(documents)

def answer_with_context(
    question: str,
    namespace: str,
    k: int
):
    """Retrieves context from Pinecone and generates an answer."""
    retrieved_docs = vectorstore.similarity_search(
        query=question, k=k, namespace=namespace
    )

    context = ""
    source_documents = []
    for doc in retrieved_docs:
        source_info = {
            "source_filename": doc.metadata.get('source_filename', 'N/A'),
            "chunk_id": doc.metadata.get('chunk_id', 'N/A'),
            "content": doc.page_content
        }
        source_documents.append(source_info)
        context += f"Source: {source_info['source_filename']} (Chunk ID: {source_info['chunk_id']})\n"
        context += f"Content: {doc.page_content}\n---\n"

    # Create the prompt
    prompt = f"""
    You are a helpful assistant. Answer the following question based on the provided context only.
    If the answer is not in the context, say "I don't have enough information to answer this question."

    Context:
    {context}

    Question: {question}

    Answer:
    """

    response = llm.invoke(prompt)
    return response.content, source_documents


# --- 4. API ENDPOINTS ---

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/status")
async def get_status():
    # A more detailed status can be provided here if needed
    return {"status": "running", "pinecone_index": index_name}

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    namespace: str = Form("files")
):
    """
    Accepts PDF and TXT files, extracts text, chunks it, and uploads to Pinecone.
    """
    filename = file.filename
    if not (filename.lower().endswith(".pdf") or filename.lower().endswith(".txt")):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only PDF and TXT files are supported."
        )

    try:
        content = await file.read()
        text_data = ""

        if filename.lower().endswith(".pdf"):
            with fitz.open(stream=content, filetype="pdf") as doc:
                text_data = "".join(page.get_text() for page in doc)
        elif filename.lower().endswith(".txt"):
            text_data = content.decode("utf-8")

        if not text_data.strip():
            raise HTTPException(status_code=400, detail="The uploaded file is empty or contains no text.")

        chunks_uploaded = upload_data_to_pinecone(
            text_data=text_data,
            source_filename=filename,
            namespace=namespace
        )

        return UploadResponse(
            filename=filename,
            message=f"Successfully uploaded and indexed '{filename}'.",
            chunks_uploaded=chunks_uploaded
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# MODIFIED: The endpoint now uses default values for namespace and k internally.
@app.post("/chat", response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest):
    """
    Receives a question and returns a RAG-generated answer.
    Uses default search parameters (namespace='files', k=4).
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Default values for namespace and k are now set here
        default_namespace = "files"
        default_k = 50

        answer, source_docs = answer_with_context(
            question=request.question,
            namespace=default_namespace,
            k=default_k
        )
        return ChatResponse(
            question=request.question,
            answer=answer,
            source_documents=source_docs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during chat processing: {str(e)}")


# --- 5. SERVER EXECUTION ---

if __name__ == "__main__":
    # To run this app: uvicorn main:app --reload
    uvicorn.run(app, host="127.0.0.1", port=8000)