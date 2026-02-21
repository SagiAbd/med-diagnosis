from __future__ import annotations

import httpx
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings

from app.core.config import settings


class TEIEmbeddings(Embeddings):
    """Embeddings via a local Text Embeddings Inference (TEI) server."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=self.timeout)
        self.async_client = httpx.AsyncClient(timeout=self.timeout)

    def _post(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.post(f"{self.base_url}/embed", json={"inputs": texts})
        resp.raise_for_status()
        return resp.json()

    async def _apost(self, texts: List[str]) -> List[List[float]]:
        resp = await self.async_client.post(f"{self.base_url}/embed", json={"inputs": texts})
        resp.raise_for_status()
        return resp.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._post([t.replace("\n", " ") for t in texts])

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self._apost([t.replace("\n", " ") for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        return (await self.aembed_documents([text]))[0]


class EmbeddingsFactory:
    @staticmethod
    def create():
        """
        Factory method to create an embeddings instance based on .env config.
        """
        embeddings_provider = settings.EMBEDDINGS_PROVIDER.lower()

        if embeddings_provider == "openai":
            return OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_API_BASE,
                model=settings.OPENAI_EMBEDDINGS_MODEL
            )
        elif embeddings_provider == "dashscope":
            return DashScopeEmbeddings(
                model=settings.DASH_SCOPE_EMBEDDINGS_MODEL,
                dashscope_api_key=settings.DASH_SCOPE_API_KEY
            )
        elif embeddings_provider == "ollama":
            return OllamaEmbeddings(
                model=settings.OLLAMA_EMBEDDINGS_MODEL,
                base_url=settings.OLLAMA_API_BASE
            )
        elif embeddings_provider == "tei":
            return TEIEmbeddings(base_url=settings.TEI_API_BASE)
        else:
            raise ValueError(f"Unsupported embeddings provider: {embeddings_provider}")
