from typing import List, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.user import User
from app.models.knowledge import KnowledgeBase, Document
from app.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    DocumentCreate,
    DocumentResponse
)
from app.api.auth import get_current_user
from app.services.document_processor import process_document

router = APIRouter()

@router.post("/", response_model=KnowledgeBaseResponse)
def create_knowledge_base(
    *,
    db: Session = Depends(get_db),
    knowledge_base_in: KnowledgeBaseCreate,
    current_user: User = Depends(get_current_user)
) -> Any:
    knowledge_base = KnowledgeBase(
        **knowledge_base_in.dict(),
        user_id=current_user.id
    )
    db.add(knowledge_base)
    db.commit()
    db.refresh(knowledge_base)
    return knowledge_base

@router.get("/", response_model=List[KnowledgeBaseResponse])
def get_knowledge_bases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100
) -> Any:
    knowledge_bases = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.user_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return knowledge_bases

@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
def get_knowledge_base(
    *,
    db: Session = Depends(get_db),
    knowledge_base_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    knowledge_base = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == current_user.id
        )
        .first()
    )
    if not knowledge_base:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return knowledge_base

@router.put("/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
def update_knowledge_base(
    *,
    db: Session = Depends(get_db),
    knowledge_base_id: int,
    knowledge_base_in: KnowledgeBaseUpdate,
    current_user: User = Depends(get_current_user)
) -> Any:
    knowledge_base = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == current_user.id
        )
        .first()
    )
    if not knowledge_base:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    for field, value in knowledge_base_in.dict(exclude_unset=True).items():
        setattr(knowledge_base, field, value)
    
    db.add(knowledge_base)
    db.commit()
    db.refresh(knowledge_base)
    return knowledge_base

@router.delete("/{knowledge_base_id}")
def delete_knowledge_base(
    *,
    db: Session = Depends(get_db),
    knowledge_base_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    knowledge_base = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == current_user.id
        )
        .first()
    )
    if not knowledge_base:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    db.delete(knowledge_base)
    db.commit()
    return {"status": "success"}

@router.post("/{knowledge_base_id}/upload", response_model=DocumentResponse)
async def upload_document(
    *,
    db: Session = Depends(get_db),
    knowledge_base_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
) -> Any:
    knowledge_base = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == current_user.id
        )
        .first()
    )
    if not knowledge_base:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Process the document using the document processor service
    document_data = await process_document(file)
    
    document = Document(
        title=document_data.title,
        content=document_data.content,
        file_path=document_data.file_path,
        knowledge_base_id=knowledge_base_id
    )
    
    db.add(document)
    db.commit()
    db.refresh(document)
    return document 