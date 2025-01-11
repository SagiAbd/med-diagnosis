from typing import List, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.core.security import get_current_user
from app.models.knowledge import KnowledgeBase, Document
from app.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    DocumentCreate,
    DocumentResponse
)
from app.services.document_processor import process_document

router = APIRouter()

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
        .filter(KnowledgeBase.user_id == current_user.id)
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
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.user_id == current_user.id
    ).first()
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
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.user_id == current_user.id
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    for field, value in kb_in.dict(exclude_unset=True).items():
        setattr(kb, field, value)

    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb

@router.post("/{kb_id}/upload", response_model=DocumentResponse)
async def upload_document(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Upload document to knowledge base.
    """
    # Check if knowledge base exists and belongs to user
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.user_id == current_user.id
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Process document
    processed_doc = await process_document(file, kb_id)

    # Create document record
    document = Document(
        title=processed_doc.title,
        file_path=processed_doc.file_path,
        file_size=processed_doc.file_size,
        content_type=processed_doc.content_type,
        knowledge_base_id=kb_id
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document

@router.delete("/{kb_id}")
def delete_knowledge_base(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Delete knowledge base.
    """
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.user_id == current_user.id
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    db.delete(kb)
    db.commit()
    return {"message": "Knowledge base deleted"}