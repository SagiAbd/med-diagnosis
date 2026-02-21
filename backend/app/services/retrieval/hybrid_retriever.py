"""HybridRetriever — unified retrieval pipeline: vector + BM25 + weighted RRF + reranker.

All retrieval goes through this single class using KB_CONFIG, which is driven
entirely by settings in config.py (env-overridable).

Weighted RRF formula (standard k=60):
    score(doc) = vector_weight / (k + rank_vec) + bm25_weight / (k + rank_bm25)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.core.config import settings
from app.services.vector_store.base import BaseVectorStore


# ─── Per-tool retrieval profiles ────────────────────────────────────

@dataclass
class RetrievalConfig:
    """Controls vector/BM25 weight balance, candidate pool size, and reranking."""
    vector_weight: float = 0.6   # contribution of cosine ranking in weighted RRF
    bm25_weight: float = 0.4     # contribution of BM25 ranking in weighted RRF
    candidate_k: int = 20        # candidates fetched from each method before fusion
    final_k: int = 5             # documents returned after fusion + optional rerank
    use_reranker: bool = True    # whether to apply cross-encoder reranking
    rrf_k: int = 60              # RRF rank-smoothing constant (standard value)


# Instantiated at import time from config so env vars take effect immediately.
KB_CONFIG = RetrievalConfig(
    vector_weight=settings.KB_VECTOR_WEIGHT,
    bm25_weight=settings.KB_BM25_WEIGHT,
    candidate_k=settings.KB_CANDIDATE_K,
    final_k=settings.KB_SEARCH_TOP_K,
    use_reranker=settings.KB_USE_RERANKER,
)


# ─── HybridRetriever ────────────────────────────────────────────────

class HybridRetriever:
    """Parameterised retrieval pipeline reused across all agent tools.

    Usage:
        retriever = HybridRetriever(vector_store, reranker=reranker)

        # lookup_service — pass registry docs so BM25 covers the full catalog
        docs = retriever.retrieve(query, LOOKUP_CONFIG,
                                  filter={"language": lang},
                                  bm25_documents=registry_docs)

        # get_service_legal_info — BM25 runs over vector candidates (legal chunks
        # per service are small; no separate doc list needed)
        docs = retriever.retrieve(query, LEGAL_CONFIG,
                                  filter={"service_id": sid, "language": lang})

        # search_knowledge_base — pass pre-loaded KB chunks for BM25
        docs = retriever.retrieve(query, KB_CONFIG,
                                  filter={"language": lang},
                                  bm25_documents=kb_docs)

    bm25_documents is optional:
    - Provided  → BM25 indexes the given document list (full coverage).
    - None      → BM25 indexes the vector search candidates (cheaper; works
                  well when the filtered set is already small, e.g. legal chunks
                  for one service_id).
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        reranker=None,
    ) -> None:
        self._vs = vector_store
        self._reranker = reranker
        self.last_timing: Dict[str, float] = {}  # populated after each retrieve() call

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig,
        filter: Optional[Dict] = None,
        bm25_documents: Optional[List[Document]] = None,
    ) -> List[Document]:
        """Run the full pipeline and return `config.final_k` documents."""
        t_total = time.perf_counter()

        # 1. Vector search — wide pool for fusion diversity.
        t0 = time.perf_counter()
        vs_results = self._vs.similarity_search_with_score(
            query,
            k=config.candidate_k * 2,
            filter=filter or None,
        )
        vector_docs: List[Document] = [doc for doc, _ in vs_results]
        t_vec = (time.perf_counter() - t0) * 1000

        # 2. BM25 — over provided documents or fall back to vector candidates.
        bm25_source = bm25_documents if bm25_documents is not None else vector_docs
        bm25_docs: List[Document] = []
        t_bm25 = 0.0
        if bm25_source:
            t0 = time.perf_counter()
            try:
                bm25 = BM25Retriever.from_documents(bm25_source)
                bm25.k = config.candidate_k
                bm25_docs = bm25.invoke(query)
            except Exception:
                bm25_docs = []  # degrade gracefully on empty / degenerate input
            t_bm25 = (time.perf_counter() - t0) * 1000

        # 3. Weighted RRF fusion.
        t0 = time.perf_counter()
        fused = self._weighted_rrf(
            vector_docs[: config.candidate_k],
            bm25_docs,
            vector_weight=config.vector_weight,
            bm25_weight=config.bm25_weight,
            rrf_k=config.rrf_k,
        )
        t_rrf = (time.perf_counter() - t0) * 1000

        # 4. Optional cross-encoder reranking.
        t_rerank = 0.0
        rerank_applied = False
        if config.use_reranker and self._reranker and fused:
            t0 = time.perf_counter()
            fused = self._rerank(query, fused)
            t_rerank = (time.perf_counter() - t0) * 1000
            rerank_applied = True

        bm25_src = "ext" if bm25_documents is not None else f"{len(vector_docs)}vc"
        self.last_timing = {
            f"vector ({config.candidate_k * 2}→{len(vector_docs)} docs)": t_vec,
            f"BM25 (src={bm25_src}, {len(bm25_docs)} hits)": t_bm25,
            f"RRF ({len(fused)} fused)": t_rrf,
            f"reranker ({'on' if rerank_applied else 'off'})": t_rerank,
        }

        return fused[: config.final_k]

    # ── Internals ──────────────────────────────────────────────────

    @staticmethod
    def _doc_id(doc: Document) -> str:
        """Stable identifier for deduplication across vector and BM25 result sets."""
        cid = doc.metadata.get("chunk_id")
        return str(cid) if cid else str(hash(doc.page_content[:200]))

    def _weighted_rrf(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        vector_weight: float,
        bm25_weight: float,
        rrf_k: int,
    ) -> List[Document]:
        """Merge two ranked lists using weighted Reciprocal Rank Fusion.

        score(doc) = vector_weight/(rrf_k+rank_vec) + bm25_weight/(rrf_k+rank_bm25)
        Documents appearing in only one list receive their partial score.
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(vector_docs):
            did = self._doc_id(doc)
            scores[did] = scores.get(did, 0.0) + vector_weight / (rrf_k + rank + 1)
            doc_map[did] = doc

        for rank, doc in enumerate(bm25_docs):
            did = self._doc_id(doc)
            scores[did] = scores.get(did, 0.0) + bm25_weight / (rrf_k + rank + 1)
            doc_map.setdefault(did, doc)

        ranked_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
        # Store the RRF score in each doc's metadata so callers (e.g. test-retrieval)
        # can surface it. The reranker may overwrite this with its own relevance_score.
        result = []
        for did in ranked_ids:
            doc = doc_map[did]
            doc.metadata["rrf_score"] = scores[did]
            result.append(doc)
        return result

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Cross-encoder reranking; falls back to RRF order on any failure."""
        try:
            return list(self._reranker.compress_documents(docs, query))
        except Exception:
            return docs

    def as_langchain_retriever(
        self,
        config: RetrievalConfig,
        filter: Optional[Dict] = None,
        bm25_documents: Optional[List[Document]] = None,
    ) -> "HybridRetrieverWrapper":
        """Return a LangChain-compatible BaseRetriever wrapping this instance.

        Useful for dropping into create_history_aware_retriever() and other
        LangChain chain builders that expect a BaseRetriever.
        """
        return HybridRetrieverWrapper(
            hybrid_retriever=self,
            config=config,
            filter_=filter,
            bm25_documents=bm25_documents,
        )


# ─── LangChain-compatible wrapper ───────────────────────────────────

class HybridRetrieverWrapper(BaseRetriever):
    """Thin BaseRetriever shim so HybridRetriever works inside LangChain chains.

    Pass to create_history_aware_retriever() or create_retrieval_chain()
    exactly as you would a plain vector-store retriever.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hybrid_retriever: HybridRetriever
    config: RetrievalConfig
    filter_: Optional[Dict] = None
    bm25_documents: Optional[List[Document]] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return self.hybrid_retriever.retrieve(
            query,
            self.config,
            filter=self.filter_,
            bm25_documents=self.bm25_documents,
        )
