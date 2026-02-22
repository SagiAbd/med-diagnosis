import os
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Medical Diagnosis"  # Project name
    VERSION: str = "0.1.0"  # Project version
    API_V1_STR: str = "/api"  # API version string

    # MySQL settings
    MYSQL_SERVER: str = os.getenv("MYSQL_SERVER", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "ragwebui")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "ragwebui")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "ragwebui")
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    @property
    def get_database_url(self) -> str:
        if self.SQLALCHEMY_DATABASE_URI:
            return self.SQLALCHEMY_DATABASE_URI
        return (
            f"mysql+mysqlconnector://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_SERVER}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
        )

    # JWT settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))

    # Chat Provider settings
    CHAT_PROVIDER: str = os.getenv("CHAT_PROVIDER", "openai")

    # Embeddings settings
    EMBEDDINGS_PROVIDER: str = os.getenv("EMBEDDINGS_PROVIDER", "tei")

    # MinIO settings
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET_NAME: str = os.getenv("MINIO_BUCKET_NAME", "documents")

    # OpenAI settings
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://hub.qazcode.ai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "oss-120b")
    OPENAI_EMBEDDINGS_MODEL: str = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-ada-002")

    # DashScope settings
    DASH_SCOPE_API_KEY: str = os.getenv("DASH_SCOPE_API_KEY", "")
    DASH_SCOPE_EMBEDDINGS_MODEL: str = os.getenv("DASH_SCOPE_EMBEDDINGS_MODEL", "")

    # Vector Store settings
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "chroma")

    # Chroma DB settings (local persistent storage)
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma")
    # Single fixed collection — all documents live here
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "documents")

    # Corpus auto-ingestion: place corpus.json at this path before starting the app
    CORPUS_JSON_PATH: str = os.getenv("CORPUS_JSON_PATH", "./data/corpus.json")
    # Full-text lookup file for parent-document retrieval (no embeddings)
    CORPUS_FULL_TEXT_PATH: str = os.getenv("CORPUS_FULL_TEXT_PATH", "./data/corpus_full_text.jsonl")
    # Default admin account created on first startup (used to own the auto-loaded KB)
    ADMIN_EMAIL:    str = os.getenv("ADMIN_EMAIL",    "admin@local")
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "changeme")

    # Qdrant DB settings
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_PREFER_GRPC: bool = os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true"

    # Deepseek settings
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_API_BASE: str = "https://api.deepseek.com/v1"  # 默认 API 地址
    DEEPSEEK_MODEL: str = "deepseek-chat"  # 默认模型名称

    # Google Gemini settings
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # Ollama settings
    OLLAMA_API_BASE: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "deepseek-r1:7b"
    OLLAMA_EMBEDDINGS_MODEL: str = os.getenv(
        "OLLAMA_EMBEDDINGS_MODEL", "nomic-embed-text"
    )

    # TEI (Text Embeddings Inference) settings
    TEI_API_BASE: str = os.getenv("TEI_API_BASE", "http://tei:80")
    TEI_EMBEDDINGS_MODEL: str = os.getenv("TEI_EMBEDDINGS_MODEL", "google/embeddinggemma-300m")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # ─── Retrieval settings ──────────────────────────────────────────────
    KB_VECTOR_WEIGHT: float = float(os.getenv("KB_VECTOR_WEIGHT", "0.6"))
    KB_BM25_WEIGHT: float = float(os.getenv("KB_BM25_WEIGHT", "0.4"))
    KB_CANDIDATE_K: int = int(os.getenv("KB_CANDIDATE_K", "20"))
    KB_SEARCH_TOP_K: int = int(os.getenv("KB_SEARCH_TOP_K", "5"))
    KB_USE_RERANKER: bool = False  # Set True/False directly here to toggle reranker
    RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "10"))

    # ─── Diagnose endpoint settings ──────────────────────────────────────
    # Default knowledge base used by the public /diagnose endpoint
    DEFAULT_KB_ID: int = int(os.getenv("DEFAULT_KB_ID", "1"))
    # How many chunks the hybrid retriever fetches before deduplication
    TEST_RETRIEVAL_CHUNK_K: int = int(os.getenv("TEST_RETRIEVAL_CHUNK_K", "20"))
    # How many unique protocols (parents) are passed to the LLM
    TEST_RETRIEVAL_PROTOCOLS_N: int = int(os.getenv("TEST_RETRIEVAL_PROTOCOLS_N", "6"))
    # Max characters per protocol text sent to the LLM (~1500 tokens each, 4×6000 = ~6k tokens context)
    DIAGNOSE_MAX_PROTOCOL_CHARS: int = int(os.getenv("DIAGNOSE_MAX_PROTOCOL_CHARS", "6000"))

    class Config:
        env_file = ".env"


settings = Settings()
