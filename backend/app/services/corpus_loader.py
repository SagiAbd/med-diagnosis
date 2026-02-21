"""
corpus_loader.py
----------------
On startup, reads ./data/corpus.json and auto-populates:
  - MySQL: users (if empty), knowledge_bases, documents, document_chunks
  - Chroma: embeds via TEI and adds to the fixed collection (skipped if
            the collection already has vectors, e.g. from the Colab script)

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

import chromadb
from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings
from app.core.security import get_password_hash
from app.db.session import SessionLocal
from app.models.knowledge import Document, DocumentChunk, KnowledgeBase
from app.models.user import User
from app.services.embedding.embedding_factory import EmbeddingsFactory
from app.services.vector_store import VectorStoreFactory

logger = logging.getLogger(__name__)

_DEFAULT_KB_NAME = "Corpus"


async def auto_ingest_corpus() -> None:
    """Idempotent startup ingestion of corpus.json into SQL + Chroma."""

    corpus_path = settings.CORPUS_JSON_PATH
    if not os.path.exists(corpus_path):
        logger.info(f"corpus.json not found at '{corpus_path}' — skipping auto-ingestion")
        return

    logger.info(f"Found corpus.json at '{corpus_path}', starting auto-ingestion…")

    with open(corpus_path, encoding="utf-8") as f:
        chunks_data: list[dict] = json.load(f)

    if not chunks_data:
        logger.warning("corpus.json is empty — skipping")
        return

    logger.info(f"Loaded {len(chunks_data)} chunks")

    db = SessionLocal()
    try:
        # ── 1. Ensure a user exists ──────────────────────────────────────────
        # Uses ADMIN_EMAIL/ADMIN_USERNAME/ADMIN_PASSWORD from config (.env).
        # If that account already exists (created via the web UI), it is reused
        # so the KB will be visible when logged in with those credentials.
        user = db.query(User).filter(User.email == settings.ADMIN_EMAIL).first()
        if user is None:
            user = db.query(User).first()   # fall back to any existing user
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
            logger.info(f"Created admin user '{settings.ADMIN_USERNAME}' (id={user.id})")

        # ── 2. Ensure the default KB exists ─────────────────────────────────
        kb = db.query(KnowledgeBase).first()
        if kb is None:
            kb = KnowledgeBase(
                name=_DEFAULT_KB_NAME,
                description="Auto-loaded from corpus.json",
                user_id=user.id,
            )
            db.add(kb)
            db.commit()
            db.refresh(kb)
            logger.info(f"Created knowledge base '{_DEFAULT_KB_NAME}' (id={kb.id})")

        # ── 3. SQL ingestion (idempotent: skip files already in DB) ─────────
        files: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks_data:
            source = chunk.get("metadata", {}).get("source", "corpus")
            files[source].append(chunk)

        for file_name, chunks in files.items():
            existing = (
                db.query(Document)
                .filter(
                    Document.file_name == file_name,
                    Document.knowledge_base_id == kb.id,
                )
                .first()
            )
            if existing:
                logger.info(f"'{file_name}' already in DB — skipping SQL insert")
                continue

            doc = Document(
                file_name=file_name,
                file_path=f"corpus/{file_name}",
                file_hash=hashlib.sha256(file_name.encode()).hexdigest(),
                file_size=sum(
                    len(c.get("page_content", "").encode()) for c in chunks
                ),
                content_type="text/plain",
                knowledge_base_id=kb.id,
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            for chunk in chunks:
                page_content = chunk.get("page_content", "")
                # Use the same ID formula as the Colab script and document_processor
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

            db.commit()
            logger.info(f"Inserted {len(chunks)} chunks for '{file_name}'")

        # ── 4. Chroma ingestion (skip if collection already has vectors) ─────
        collection_name = settings.CHROMA_COLLECTION_NAME
        chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

        has_vectors = False
        try:
            col = chroma_client.get_collection(collection_name)
            has_vectors = col.count() > 0
        except Exception:
            pass

        if has_vectors:
            logger.info(
                f"Chroma collection '{collection_name}' already has vectors "
                "(from pre-computed Colab data) — skipping embedding"
            )
            return

        logger.info(
            f"Chroma collection '{collection_name}' is empty — "
            f"embedding {len(chunks_data)} chunks via TEI…"
        )

        embeddings = EmbeddingsFactory.create()
        vector_store = VectorStoreFactory.create(
            store_type=settings.VECTOR_STORE_TYPE,
            collection_name=collection_name,
            embedding_function=embeddings,
        )

        langchain_docs: list[LangchainDocument] = []
        for chunk in chunks_data:
            page_content = chunk.get("page_content", "")
            metadata = dict(chunk.get("metadata", {}))
            file_name = metadata.get("source", "corpus")
            chunk_id = hashlib.sha256(
                f"{file_name}:{page_content}".encode()
            ).hexdigest()
            metadata["chunk_id"] = chunk_id
            langchain_docs.append(
                LangchainDocument(page_content=page_content, metadata=metadata)
            )

        batch_size = 100
        total_batches = (len(langchain_docs) + batch_size - 1) // batch_size
        for i in range(0, len(langchain_docs), batch_size):
            batch = langchain_docs[i : i + batch_size]
            vector_store.add_documents(batch)
            logger.info(f"Embedded batch {i // batch_size + 1}/{total_batches}")

        logger.info("Corpus auto-ingestion complete")

    except Exception:
        logger.exception("Corpus auto-ingestion failed")
    finally:
        db.close()
