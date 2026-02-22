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

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LCDocument

from app.core.config import settings
from app.core.security import get_password_hash
from app.db.session import SessionLocal
from app.models.knowledge import Document, DocumentChunk, KnowledgeBase
from app.models.user import User

logger = logging.getLogger(__name__)

_DEFAULT_KB_NAME = "Corpus"

# Module-level lookup: protocol_id → full protocol text.
# Populated at startup from corpus_full_text.jsonl; read-only afterwards.
protocol_full_text_store: dict[str, str] = {}

# Module-level lookup: protocol_id → list of ICD-10 codes (may be empty).
# Populated at startup from corpus_full_text.jsonl alongside protocol_full_text_store.
protocol_icd_codes_store: dict[str, list[str]] = {}

# Pre-built BM25 retriever over the full protocol corpus.
# Populated at startup by _load_protocol_full_text(); None until then.
# Exposed so knowledge_base.py can pass it to HybridRetriever once at startup.
protocol_bm25_retriever = None  # BM25Retriever | None


def _load_protocol_full_text() -> None:
    """Load corpus_full_text.jsonl into protocol_full_text_store and build BM25 index."""
    global protocol_bm25_retriever
    path = settings.CORPUS_FULL_TEXT_PATH
    if not os.path.exists(path):
        logger.info(f"corpus_full_text.jsonl not found at '{path}' — skipping full-text load")
        return
    store: dict[str, str] = {}
    icd_store: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get("protocol_id")
                text = obj.get("text", "")
                icd_codes = obj.get("icd_codes", [])
                if pid:
                    store[pid] = text
                    icd_store[pid] = icd_codes if isinstance(icd_codes, list) else []
            except json.JSONDecodeError:
                continue
    protocol_full_text_store.update(store)  # mutate in-place so all importers see the data
    protocol_icd_codes_store.update(icd_store)
    logger.info(f"Loaded {len(store)} full-text protocols from '{path}'")

    # Build corpus-level BM25 index from full protocol texts.
    # Imported here (not at module top) to avoid circular imports at load time.
    if store:
        try:
            from app.services.retrieval.hybrid_retriever import russian_preprocess
            bm25_docs = [
                LCDocument(page_content=text, metadata={"protocol_id": pid})
                for pid, text in store.items()
            ]
            bm25 = BM25Retriever.from_documents(
                bm25_docs, preprocess_func=russian_preprocess
            )
            bm25.k = 50  # callers slice to their own candidate_k
            protocol_bm25_retriever = bm25
            logger.info(
                f"Built BM25 corpus index over {len(bm25_docs)} protocols "
                f"(k=50, Russian preprocessor)"
            )
        except Exception:
            logger.exception("Failed to build BM25 corpus index — BM25 will fall back to vector candidates")


async def auto_ingest_corpus() -> None:
    """Full replace of SQL + Chroma from corpus.json on every startup."""

    # Always load full-text lookup regardless of corpus.json presence.
    _load_protocol_full_text()

    # ── Always ensure admin user exists (independent of corpus.json) ─────────
    db = SessionLocal()
    try:
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
    except Exception:
        logger.exception("Failed to ensure admin user exists")
    finally:
        db.close()

    corpus_path = settings.CORPUS_JSON_PATH
    if not os.path.exists(corpus_path):
        logger.info(f"corpus.json not found at '{corpus_path}' — skipping")
        return

    logger.info(f"Found corpus.json at '{corpus_path}', starting full re-ingestion…")

    with open(corpus_path, encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        logger.warning(
            f"corpus.json at '{corpus_path}' is empty (possibly an undownloaded Git LFS pointer). "
            "Run 'git lfs pull' to fetch the actual file. Skipping corpus ingestion."
        )
        return

    try:
        chunks_data: list[dict] = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(
            f"corpus.json could not be parsed as JSON ({e}). "
            "It may be a Git LFS pointer file — run 'git lfs pull' to download the actual data. "
            "Skipping corpus ingestion."
        )
        return

    if not chunks_data:
        logger.warning("corpus.json is empty — skipping")
        return

    logger.info(f"Loaded {len(chunks_data)} chunks")

    db = SessionLocal()
    try:
        # ── 1. Get admin user for KB ownership ───────────────────────────────
        user = db.query(User).filter(User.email == settings.ADMIN_EMAIL).first()
        if user is None:
            user = db.query(User).first()

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

            BATCH = 500
            for i, chunk in enumerate(chunks):
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
                if (i + 1) % BATCH == 0:
                    db.commit()
                    logger.info(f"  committed {i + 1}/{len(chunks)} chunks for '{file_name}'")

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
