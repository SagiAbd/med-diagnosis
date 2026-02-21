import hashlib
import json
from typing import List, Any, Dict
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query
from sqlalchemy.orm import Session
from langchain_chroma import Chroma
from sqlalchemy import text
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel
from sqlalchemy.orm import selectinload
import time
import asyncio

from app.db.session import get_db
from app.models.user import User
from app.core.security import get_current_user
from app.models.knowledge import KnowledgeBase, Document, ProcessingTask, DocumentChunk, DocumentUpload
from app.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    DocumentResponse,
    PreviewRequest
)
from app.services.document_processor import process_document_background, upload_document, preview_document, PreviewResult
from app.core.config import settings
from app.core.minio import get_minio_client
from minio.error import MinioException
from app.services.vector_store import VectorStoreFactory
from app.services.embedding.embedding_factory import EmbeddingsFactory
from app.services.retrieval import HybridRetriever, RetrievalConfig
from app.services.llm.llm_factory import LLMFactory
from app.services.corpus_loader import protocol_full_text_store
from langchain_core.messages import HumanMessage

router = APIRouter()

logger = logging.getLogger(__name__)

class TestRetrievalRequest(BaseModel):
    query: str
    kb_id: int


class DiagnosisItem(BaseModel):
    rank: int
    diagnosis: str
    icd10_code: str
    explanation: str


class DiagnosisResponse(BaseModel):
    diagnoses: List[DiagnosisItem]

@router.post("", response_model=KnowledgeBaseResponse)
def create_knowledge_base(
    *,
    db: Session = Depends(get_db),
    kb_in: KnowledgeBaseCreate,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Create new knowledge base.
    """
    kb = KnowledgeBase(
        name=kb_in.name,
        description=kb_in.description,
        user_id=current_user.id
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    logger.info(f"Knowledge base created: {kb.name} for user {current_user.id}")
    return kb

@router.get("", response_model=List[KnowledgeBaseResponse])
def get_knowledge_bases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100
) -> Any:
    """
    Retrieve knowledge bases.
    """
    knowledge_bases = (
        db.query(KnowledgeBase)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return knowledge_bases

@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
def get_knowledge_base(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get knowledge base by ID.
    """
    from sqlalchemy.orm import joinedload
    
    kb = (
        db.query(KnowledgeBase)
        .options(
            joinedload(KnowledgeBase.documents)
            .joinedload(Document.processing_tasks)
        )
        .filter(KnowledgeBase.id == kb_id)
        .first()
    )

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    return kb

@router.put("/{kb_id}", response_model=KnowledgeBaseResponse)
def update_knowledge_base(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    kb_in: KnowledgeBaseUpdate,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Update knowledge base.
    """
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    for field, value in kb_in.dict(exclude_unset=True).items():
        setattr(kb, field, value)

    db.add(kb)
    db.commit()
    db.refresh(kb)
    logger.info(f"Knowledge base updated: {kb.name} for user {current_user.id}")
    return kb

@router.delete("/{kb_id}")
async def delete_knowledge_base(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Delete knowledge base and all associated resources.
    """
    logger = logging.getLogger(__name__)
    
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    try:
        # Get all document file paths before deletion
        document_paths = [doc.file_path for doc in kb.documents]
        
        # Initialize services
        minio_client = get_minio_client()
        embeddings = EmbeddingsFactory.create()

        vector_store = VectorStoreFactory.create(
            store_type=settings.VECTOR_STORE_TYPE,
            collection_name=f"kb_{kb_id}",
            embedding_function=embeddings,
        )
        
        # Clean up external resources first
        cleanup_errors = []
        
        # 1. Clean up MinIO files
        try:
            # Delete all objects with prefix kb_{kb_id}/
            objects = minio_client.list_objects(settings.MINIO_BUCKET_NAME, prefix=f"kb_{kb_id}/")
            for obj in objects:
                minio_client.remove_object(settings.MINIO_BUCKET_NAME, obj.object_name)
            logger.info(f"Cleaned up MinIO files for knowledge base {kb_id}")
        except MinioException as e:
            cleanup_errors.append(f"Failed to clean up MinIO files: {str(e)}")
            logger.error(f"MinIO cleanup error for kb {kb_id}: {str(e)}")
        
        # 2. Clean up vector store
        try:
            vector_store._store.delete_collection(f"kb_{kb_id}")
            logger.info(f"Cleaned up vector store for knowledge base {kb_id}")
        except Exception as e:
            cleanup_errors.append(f"Failed to clean up vector store: {str(e)}")
            logger.error(f"Vector store cleanup error for kb {kb_id}: {str(e)}")
        
        # Finally, delete database records in a single transaction
        db.delete(kb)
        db.commit()
        
        # Report any cleanup errors in the response
        if cleanup_errors:
            return {
                "message": "Knowledge base deleted with cleanup warnings",
                "warnings": cleanup_errors
            }
        
        return {"message": "Knowledge base and all associated resources deleted successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete knowledge base {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete knowledge base: {str(e)}")

# Batch upload documents
@router.post("/{kb_id}/documents/upload")
async def upload_kb_documents(
    kb_id: int,
    files: List[UploadFile],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload multiple documents to MinIO.
    """
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    results = []
    for file in files:
        # 1. 计算文件 hash
        file_content = await file.read()
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # 2. 检查是否存在完全相同的文件（名称和hash都相同）
        existing_document = db.query(Document).filter(
            Document.file_name == file.filename,
            Document.file_hash == file_hash,
            Document.knowledge_base_id == kb_id
        ).first()
        
        if existing_document:
            # 完全相同的文件，直接返回
            results.append({
                "document_id": existing_document.id,
                "file_name": existing_document.file_name,
                "status": "exists",
                "message": "文件已存在且已处理完成",
                "skip_processing": True
            })
            continue
        
        # 3. 上传到临时目录
        temp_path = f"kb_{kb_id}/temp/{file.filename}"
        await file.seek(0)
        try:
            minio_client = get_minio_client()
            file_size = len(file_content)  # 使用之前读取的文件内容长度
            minio_client.put_object(
                bucket_name=settings.MINIO_BUCKET_NAME,
                object_name=temp_path,
                data=file.file,
                length=file_size,  # 指定文件大小
                content_type=file.content_type
            )
        except MinioException as e:
            logger.error(f"Failed to upload file to MinIO: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to upload file")
        
        # 4. 创建上传记录
        upload = DocumentUpload(
            knowledge_base_id=kb_id,
            file_name=file.filename,
            file_hash=file_hash,
            file_size=len(file_content),
            content_type=file.content_type,
            temp_path=temp_path
        )
        db.add(upload)
        db.commit()
        db.refresh(upload)
        
        results.append({
            "upload_id": upload.id,
            "file_name": file.filename,
            "temp_path": temp_path,
            "status": "pending",
            "skip_processing": False
        })
    
    return results

@router.post("/{kb_id}/documents/preview")
async def preview_kb_documents(
    kb_id: int,
    preview_request: PreviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[int, PreviewResult]:
    """
    Preview multiple documents' chunks.
    """
    results = {}
    for doc_id in preview_request.document_ids:
        document = db.query(Document).filter(
            Document.id == doc_id,
            Document.knowledge_base_id == kb_id,
        ).first()

        if document:
            file_path = document.file_path
        else:
            upload = db.query(DocumentUpload).filter(
                DocumentUpload.id == doc_id,
                DocumentUpload.knowledge_base_id == kb_id,
            ).first()
            
            if not upload:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
            
            file_path = upload.temp_path
        
        preview = await preview_document(
            file_path,
            chunk_size=preview_request.chunk_size,
            chunk_overlap=preview_request.chunk_overlap
        )
        results[doc_id] = preview
    
    return results

@router.post("/{kb_id}/documents/process")
async def process_kb_documents(
    kb_id: int,
    upload_results: List[dict],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Process multiple documents asynchronously.
    """
    start_time = time.time()
    
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    task_info = []
    upload_ids = []
    
    for result in upload_results:
        if result.get("skip_processing"):
            continue
        upload_ids.append(result["upload_id"])
    
    if not upload_ids:
        return {"tasks": []}
    
    uploads = db.query(DocumentUpload).filter(DocumentUpload.id.in_(upload_ids)).all()
    uploads_dict = {upload.id: upload for upload in uploads}
    
    all_tasks = []
    for upload_id in upload_ids:
        upload = uploads_dict.get(upload_id)
        if not upload:
            continue
            
        task = ProcessingTask(
            document_upload_id=upload_id,
            knowledge_base_id=kb_id,
            status="pending"
        )
        all_tasks.append(task)
    
    db.add_all(all_tasks)
    db.commit()
    
    for task in all_tasks:
        db.refresh(task)
    
    task_data = []
    for i, upload_id in enumerate(upload_ids):
        if i < len(all_tasks):
            task = all_tasks[i]
            upload = uploads_dict.get(upload_id)
            
            task_info.append({
                "upload_id": upload_id,
                "task_id": task.id
            })
            
            if upload:
                task_data.append({
                    "task_id": task.id,
                    "upload_id": upload_id,
                    "temp_path": upload.temp_path,
                    "file_name": upload.file_name
                })
    
    background_tasks.add_task(
        add_processing_tasks_to_queue,
        task_data,
        kb_id
    )
    
    return {"tasks": task_info}

async def add_processing_tasks_to_queue(task_data, kb_id):
    """Helper function to add document processing tasks to the queue without blocking the main response."""
    for data in task_data:
        asyncio.create_task(
            process_document_background(
                data["temp_path"],
                data["file_name"],
                kb_id,
                data["task_id"],
                None
            )
        )
    logger.info(f"Added {len(task_data)} document processing tasks to queue")

@router.post("/cleanup")
async def cleanup_temp_files(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Clean up expired temporary files.
    """
    expired_time = datetime.utcnow() - timedelta(hours=24)
    expired_uploads = db.query(DocumentUpload).filter(
        DocumentUpload.created_at < expired_time
    ).all()
    
    minio_client = get_minio_client()
    for upload in expired_uploads:
        try:
            minio_client.remove_object(
                bucket_name=settings.MINIO_BUCKET_NAME,
                object_name=upload.temp_path
            )
        except MinioException as e:
            logger.error(f"Failed to delete temp file {upload.temp_path}: {str(e)}")
        
        db.delete(upload)
    
    db.commit()
    
    return {"message": f"Cleaned up {len(expired_uploads)} expired uploads"}

@router.get("/{kb_id}/documents/tasks")
async def get_processing_tasks(
    kb_id: int,
    task_ids: str = Query(..., description="Comma-separated list of task IDs to check status for"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get status of multiple processing tasks.
    """
    task_id_list = [int(id.strip()) for id in task_ids.split(",")]
    
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    tasks = (
        db.query(ProcessingTask)
        .options(
            selectinload(ProcessingTask.document_upload)
        )
        .filter(
            ProcessingTask.id.in_(task_id_list),
            ProcessingTask.knowledge_base_id == kb_id
        )
        .all()
    )
    
    return {
        task.id: {
            "document_id": task.document_id,
            "status": task.status,
            "error_message": task.error_message,
            "upload_id": task.document_upload_id,
            "file_name": task.document_upload.file_name if task.document_upload else None
        }
        for task in tasks
    }

@router.get("/{kb_id}/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    doc_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get document details by ID.
    """
    document = (
        db.query(Document)
        .filter(
            Document.id == doc_id,
            Document.knowledge_base_id == kb_id,
        )
        .first()
    )

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document

@router.post("/test-retrieval")
async def test_retrieval(
    request: TestRetrievalRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Diagnose a patient query using hybrid retrieval + full-protocol context + LLM.
    Returns a ranked list of diagnoses with ICD-10 codes extracted from protocol text.
    """
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == request.kb_id).first()
        if not kb:
            raise HTTPException(status_code=404, detail=f"Knowledge base {request.kb_id} not found")

        # 1. Hybrid retrieval — fetch top TEST_RETRIEVAL_CHUNK_K chunks
        embeddings = EmbeddingsFactory.create()
        vector_store = VectorStoreFactory.create(
            store_type=settings.VECTOR_STORE_TYPE,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
        )
        test_config = RetrievalConfig(
            vector_weight=settings.KB_VECTOR_WEIGHT,
            bm25_weight=settings.KB_BM25_WEIGHT,
            candidate_k=settings.KB_CANDIDATE_K,
            final_k=settings.TEST_RETRIEVAL_CHUNK_K,
            use_reranker=settings.KB_USE_RERANKER,
        )
        retriever = HybridRetriever(vector_store)
        chunks = retriever.retrieve(request.query, test_config)

        # 2. Deduplicate by protocol_id — collect top N unique parent protocols
        seen_protocol_ids: list[str] = []
        for chunk in chunks:
            pid = chunk.metadata.get("protocol_id")
            if pid and pid not in seen_protocol_ids:
                seen_protocol_ids.append(pid)
            if len(seen_protocol_ids) >= settings.TEST_RETRIEVAL_PROTOCOLS_N:
                break

        if not seen_protocol_ids:
            raise HTTPException(status_code=404, detail="No protocols found for the given query")

        # 3. Fetch full protocol texts (fall back to chunk content if full text missing)
        protocol_sections: list[str] = []
        for i, pid in enumerate(seen_protocol_ids, 1):
            full_text = protocol_full_text_store.get(pid, "")
            if full_text:
                protocol_sections.append(f"=== Протокол {i} (ID: {pid}) ===\n{full_text}")
            else:
                # Fallback: use the first matching chunk
                for chunk in chunks:
                    if chunk.metadata.get("protocol_id") == pid:
                        protocol_sections.append(
                            f"=== Протокол {i} (ID: {pid}) ===\n{chunk.page_content}"
                        )
                        break

        context = "\n\n".join(protocol_sections)
        n = len(seen_protocol_ids)

        # 4. LLM call — structured JSON diagnosis
        prompt = (
            "Вы — медицинский ассистент. На основе анамнеза пациента и предоставленных "
            "клинических протоколов Республики Казахстан определите наиболее вероятные диагнозы.\n\n"
            f"Анамнез пациента:\n{request.query}\n\n"
            f"Клинические протоколы:\n{context}\n\n"
            "Верните ответ СТРОГО в формате JSON без каких-либо дополнительных комментариев:\n"
            '{\n    "diagnoses": [\n'
            '        {\n'
            '            "rank": 1,\n'
            '            "diagnosis": "название диагноза",\n'
            '            "icd10_code": "код МКБ-10 из протокола",\n'
            '            "explanation": "краткое обоснование на основе симптомов и протокола"\n'
            "        }\n"
            "    ]\n"
            "}\n\n"
            f"Верните ровно {n} диагноз(а/ов), ранжированных по убыванию вероятности. "
            "Код МКБ-10 извлеките из текста соответствующего протокола."
        )

        llm = LLMFactory.create(temperature=0, streaming=False)
        response = await llm.ainvoke([HumanMessage(content=prompt)])

        # 5. Parse and validate with Pydantic
        raw = response.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        validated = DiagnosisResponse(**parsed)

        return validated.model_dump()

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"LLM returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
