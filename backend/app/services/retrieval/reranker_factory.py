"""RerankerFactory — creates a cross-encoder reranker for HybridRetriever.

FlashRank is used by default: it runs on CPU, has ~5 ms latency for 20 docs,
and requires no GPU — appropriate for a local model deployment.

The reranker returned is a LangChain document compressor that implements
    compress_documents(documents, query) -> List[Document]

so HybridRetriever._rerank() can call it without knowing the implementation.

To swap to a different reranker (e.g. Cohere, local cross-encoder):
    replace the body of create_reranker() — the rest of the system is unchanged.
"""

from __future__ import annotations

from app.core.config import settings


def create_reranker(top_n: int | None = None):
    """Return a FlashRank reranker compressor.

    Args:
        top_n: Maximum documents to keep after reranking.  Defaults to
               settings.RERANKER_TOP_N so the value is env-configurable.
               Should be >= max(LEGAL_CONFIG.final_k, KB_CONFIG.final_k);
               HybridRetriever applies config.final_k as the final slice.

    Returns:
        FlashrankRerank instance, or None if flashrank is not installed.
        HybridRetriever degrades gracefully when reranker is None (keeps
        RRF order and skips the reranking step).
    """
    if top_n is None:
        top_n = settings.RERANKER_TOP_N

    try:
        from flashrank import Ranker  # must be imported first to resolve the forward ref
        from langchain_community.document_compressors import FlashrankRerank
        FlashrankRerank.model_rebuild()  # resolve Pydantic v2 forward reference to Ranker
        return FlashrankRerank(top_n=top_n)
    except ImportError:
        print(
            "[RerankerFactory] flashrank not installed — reranking disabled. "
            "Install with: pip install flashrank"
        )
        return None
