import os
from typing import Optional
from fastapi import UploadFile
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from app.core.config import settings

class ProcessedDocument(BaseModel):
    title: str
    content: str
    file_path: Optional[str] = None
    file_size: int
    content_type: str

async def process_document(file: UploadFile, kb_id: int) -> ProcessedDocument:
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    # Save the file
    file_path = f"uploads/{file.filename}"
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Get file extension
    _, ext = os.path.splitext(file.filename)
    ext = ext.lower()
    
    # Select appropriate loader
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        content_type = "application/pdf"
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
        content_type = "text/markdown"
    else:  # Default to text loader
        loader = TextLoader(file_path)
        content_type = "text/plain"
    
    # Load and split the document
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE
    )
    
    # Store document chunks in ChromaDB
    vector_store = Chroma(
        collection_name=f"kb_{kb_id}",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Add documents to vector store
    vector_store.add_documents(chunks)
    
    # Combine all chunks into a single text for the database record
    combined_text = "\n\n".join([chunk.page_content for chunk in chunks])
    
    return ProcessedDocument(
        title=file.filename,
        content=combined_text,
        file_path=file_path,
        file_size=len(content),
        content_type=content_type
    ) 