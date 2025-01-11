from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import relationship
from app.models.base import Base, TimestampMixin

class KnowledgeBase(Base, TimestampMixin):
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(LONGTEXT)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    documents = relationship("Document", back_populates="knowledge_base", cascade="all, delete-orphan")
    user = relationship("User", back_populates="knowledge_bases")

class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)  # Path in MinIO
    file_size = Column(Integer)  # File size in bytes
    content_type = Column(String(100))  # MIME type
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False)
    
    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="documents") 