"""RerankerFactory — creates a cross-encoder reranker for HybridRetriever.

Uses sentence-transformers HuggingFaceCrossEncoder via LangChain's CrossEncoderReranker.
Cross-encoders jointly encode the query and document (unlike bi-encoders) which gives
more accurate relevance scores at the cost of being slower.

Default model: BAAI/bge-reranker-v2-m3 — multilingual, handles Russian well.
Override via RERANKER_MODEL env var.

The reranker returned implements:
    compress_documents(documents, query) -> List[Document]
so HybridRetriever._rerank() can call it without knowing the implementation.
"""

from __future__ import annotations

from app.core.config import settings


def create_reranker(top_n: int | None = None):
    """Return a CrossEncoderReranker document compressor.

    Args:
        top_n: Maximum documents to keep after reranking.  Defaults to
               settings.RERANKER_TOP_N so the value is env-configurable.
               HybridRetriever applies config.final_k as the final slice.

    Returns:
        CrossEncoderReranker, or None if sentence-transformers is not installed.
        HybridRetriever degrades gracefully when reranker is None (keeps
        RRF order and skips the reranking step).
    """
    if top_n is None:
        top_n = settings.RERANKER_TOP_N

    try:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain.retrievers.document_compressors import CrossEncoderReranker

        model = HuggingFaceCrossEncoder(model_name=settings.RERANKER_MODEL)
        return CrossEncoderReranker(model=model, top_n=top_n)
    except ImportError as e:
        print(
            f"[RerankerFactory] Missing dependency ({e}) — reranking disabled. "
            "Install with: pip install sentence-transformers"
        )
        return None
    except Exception as e:
        print(
            f"[RerankerFactory] Failed to load reranker '{settings.RERANKER_MODEL}': {e} "
            "— reranking disabled."
        )
        return None
