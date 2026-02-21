"""
corpus_loader.py
----------------
On every startup, fully replaces SQL with the contents of corpus.json:
  1. Wipe existing DocumentChunk / Document / KnowledgeBase rows for the corpus KB
  2. Re-insert all records from corpus.json
  Chroma is left untouched — embeddings are pre-computed and stored in chroma_data/.

corpus.json format  (flat array of LangChain-style dicts):
  [
    {"page_content": "...", "metadata": {"source": "file.pdf", "page": 0, ...}},
    ...
  ]
"""
import hashlib
import json
import logging
import os
from collections import defaultdict

from app.core.config import settings
from app.core.security import get_password_hash
from app.db.session import SessionLocal
from app.models.knowledge import Document, DocumentChunk, KnowledgeBase
from app.models.user import User

logger = logging.getLogger(__name__)

_DEFAULT_KB_NAME = "Corpus"


async def auto_ingest_corpus() -> None:
    """Full replace of SQL + Chroma from corpus.json on every startup."""

    corpus_path = settings.CORPUS_JSON_PATH
    if not os.path.exists(corpus_path):
        logger.info(f"corpus.json not found at '{corpus_path}' — skipping")
        return

    logger.info(f"Found corpus.json at '{corpus_path}', starting full re-ingestion…")

    with open(corpus_path, encoding="utf-8") as f:
        chunks_data: list[dict] = json.load(f)

    if not chunks_data:
        logger.warning("corpus.json is empty — skipping")
        return

    logger.info(f"Loaded {len(chunks_data)} chunks")

    db = SessionLocal()
    try:
        # ── 1. Ensure admin user exists ──────────────────────────────────────
        user = db.query(User).filter(User.email == settings.ADMIN_EMAIL).first()
        if user is None:
            user = db.query(User).first()
        if user is None:
            user = User(
                email=settings.ADMIN_EMAIL,
                username=settings.ADMIN_USERNAME,
                hashed_password=get_password_hash(settings.ADMIN_PASSWORD),
                is_active=True,
                is_superuser=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"Created admin user '{settings.ADMIN_USERNAME}'")

        # ── 2. Wipe existing KB (cascades to documents + chunks) ─────────────
        existing_kb = db.query(KnowledgeBase).first()
        if existing_kb is not None:
            # Delete chunks first (FK constraint)
            db.query(DocumentChunk).filter(DocumentChunk.kb_id == existing_kb.id).delete()
            db.query(Document).filter(Document.knowledge_base_id == existing_kb.id).delete()
            db.delete(existing_kb)
            db.commit()
            logger.info("Wiped existing KB, documents and chunks from SQL")

        # ── 3. Create fresh KB ───────────────────────────────────────────────
        kb = KnowledgeBase(
            name=_DEFAULT_KB_NAME,
            description="Auto-loaded from corpus.json",
            user_id=user.id,
        )
        db.add(kb)
        db.commit()
        db.refresh(kb)
        logger.info(f"Created KB '{_DEFAULT_KB_NAME}' (id={kb.id})")

        # ── 4. Insert all documents + chunks ────────────────────────────────
        files: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks_data:
            source = chunk.get("metadata", {}).get("source", "corpus")
            files[source].append(chunk)

        total_chunks = 0
        for file_name, chunks in files.items():
            doc = Document(
                file_name=file_name,
                file_path=f"corpus/{file_name}",
                file_hash=hashlib.sha256(file_name.encode()).hexdigest(),
                file_size=sum(len(c.get("page_content", "").encode()) for c in chunks),
                content_type="text/plain",
                knowledge_base_id=kb.id,
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            for chunk in chunks:
                page_content = chunk.get("page_content", "")
                chunk_id = hashlib.sha256(
                    f"{file_name}:{page_content}".encode()
                ).hexdigest()
                metadata = dict(chunk.get("metadata", {}))
                metadata.update(
                    source=file_name,
                    kb_id=kb.id,
                    document_id=doc.id,
                    chunk_id=chunk_id,
                )
                chunk_hash = hashlib.sha256(
                    (page_content + str(metadata)).encode()
                ).hexdigest()
                db.add(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=doc.id,
                        kb_id=kb.id,
                        file_name=file_name,
                        chunk_metadata={"page_content": page_content, **metadata},
                        hash=chunk_hash,
                    )
                )
                total_chunks += 1

            db.commit()
            logger.info(f"Inserted {len(chunks)} chunks for '{file_name}'")

        logger.info(f"SQL: inserted {total_chunks} chunks total")
        logger.info("Chroma left untouched — embeddings loaded from chroma_data/")
        logger.info("Corpus re-ingestion complete")

    except Exception:
        logger.exception("Corpus auto-ingestion failed")
        db.rollback()
    finally:
        db.close()
