

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

load_dotenv()
app = FastAPI(
    title="RAG API with FastAPI",
    description="An API for uploading documents and chatting with a RAG model.",
    version="1.0.0",
)

origins = [
    "http://localhost:3000",
    "https://insure-rag.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    # ✅ USE THE ROBUST REGEX INSTEAD
    allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:3000",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "insurance-rag-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
pinecone_index = pc.Index(index_name)
print("Pinecone index is ready.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embeddings, namespace="files")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=os.getenv("GOOGLE_API_KEY"))


# --- 2. Pydantic Models for Request/Response ---

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

# --- NEW HELPER FUNCTION FOR VALIDATION ---
def is_insurance_related(text: str) -> bool:
    """
    Uses the LLM to quickly classify if the text is related to insurance.
    Returns True if the text is about insurance, False otherwise.
    """
    # We only need the beginning of the text to determine its topic.
    # This saves tokens and speeds up the check.
    text_snippet = text[:3000]

    # A specific, constrained prompt for a clear yes/no answer.
    prompt = f"""
    Analyze the following text. Is the primary topic of this text related to insurance, such as an insurance policy, claims, coverage details, or benefits?
    Please answer with only a single word: 'yes' or 'no'.

    Text:
    ---
    {text_snippet}
    ---
    Answer:
    """
    try:
        response = llm.invoke(prompt)
        # Clean up the response to handle whitespace and capitalization
        answer = response.content.strip().lower()
        print(f"Insurance validation check for document returned: '{answer}'")
        return answer == "yes"
    except Exception as e:
        print(f"An error occurred during insurance validation: {e}")
        # In case of an error, default to allowing the upload to avoid blocking valid files.
        return True

def upload_data_to_pinecone(text_data: str, source_filename: str, namespace: str):
    """Chunks text data and uploads it to Pinecone."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text_data)
    documents = [Document(page_content=chunk, metadata={"source_filename": source_filename, "chunk_id": i}) for i, chunk in enumerate(chunks)]
    vectorstore.add_documents(documents=documents, namespace=namespace)
    return len(documents)

def answer_with_context(question: str, namespace: str, k: int):
    """Retrieves context from Pinecone and generates an answer."""
    retrieved_docs = vectorstore.similarity_search(query=question, k=k, namespace=namespace)
    context, source_documents = "", []
    for doc in retrieved_docs:
        source_info = {
            "source_filename": doc.metadata.get('source_filename', 'N/A'),
            "chunk_id": doc.metadata.get('chunk_id', 'N/A'),
            "content": doc.page_content
        }
        source_documents.append(source_info)
        context += f"Source: {source_info['source_filename']} (Chunk ID: {source_info['chunk_id']})\nContent: {doc.page_content}\n---\n"
    prompt = f"""You are a helpful assistant. Answer the following question based on the provided context only. If the answer is not in the context, say "I don't have enough information to answer this question."\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
    response = llm.invoke(prompt)
    return response.content, source_documents


# --- 4. API ENDPOINTS ---

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- MODIFIED /upload ENDPOINT ---
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), namespace: str = Form("files")):
    """
    Accepts, validates, and processes PDF/TXT files for the insurance RAG system.
    """
    filename = file.filename
    if not (filename.lower().endswith(".pdf") or filename.lower().endswith(".txt")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and TXT files are supported.")

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

        # --- NEW VALIDATION STEP ---
        # Check if the document content is related to insurance before processing.
        if not is_insurance_related(text_data):
            raise HTTPException(
                status_code=400,
                detail="The document does not appear to be related to insurance. Please upload a relevant file."
            )
        # --- END OF VALIDATION STEP ---

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

    except HTTPException as http_exc:
        # Re-raise HTTPException to ensure FastAPI handles it correctly
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest):
    """Receives a question and returns a RAG-generated answer."""
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer, source_docs = answer_with_context(question=request.question, namespace="files", k=50)
        return ChatResponse(question=request.question, answer=answer, source_documents=source_docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during chat processing: {str(e)}")


# --- 5. SERVER EXECUTION ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
