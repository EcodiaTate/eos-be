"""
EcodiaOS — Embedding Client Abstraction

All embeddings in EOS are 768-dimensional.
Supports local sentence-transformers, API-based, and sidecar models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx
import numpy as np
import structlog

from ecodiaos.config import EmbeddingConfig

logger = structlog.get_logger()


class EmbeddingClient(ABC):
    """Abstract interface for text embedding."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text. Returns a vector of configured dimension."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...


class LocalEmbeddingClient(EmbeddingClient):
    """
    Local embedding using sentence-transformers.
    Loaded in-process. Best for latency and privacy.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device=self._device)
            logger.info(
                "embedding_model_loaded",
                model=self._model_name,
                device=self._device,
                dimension=self._model.get_sentence_embedding_dimension(),
            )

    async def embed(self, text: str) -> list[float]:
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True, batch_size=32)
        return embeddings.tolist()

    async def close(self) -> None:
        self._model = None


class SidecarEmbeddingClient(EmbeddingClient):
    """
    Embedding via HTTP sidecar service.
    For when you want the model in a separate process/container.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, text: str) -> list[float]:
        response = await self._client.post(
            f"{self._url}/embed",
            json={"text": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.post(
            f"{self._url}/embed_batch",
            json={"texts": texts},
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    async def close(self) -> None:
        await self._client.aclose()


class MockEmbeddingClient(EmbeddingClient):
    """
    Mock embedding client for testing and development.
    Returns random normalised vectors of the correct dimension.
    """

    def __init__(self, dimension: int = 768) -> None:
        self._dimension = dimension

    async def embed(self, text: str) -> list[float]:
        vec = np.random.randn(self._dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = []
        for _ in texts:
            vec = np.random.randn(self._dimension).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            results.append(vec.tolist())
        return results

    async def close(self) -> None:
        pass


def create_embedding_client(config: EmbeddingConfig) -> EmbeddingClient:
    """Factory to create the configured embedding client."""
    if config.strategy == "local":
        return LocalEmbeddingClient(
            model_name=config.local_model,
            device=config.local_device,
        )
    elif config.strategy == "sidecar":
        if not config.sidecar_url:
            raise ValueError("Sidecar strategy requires sidecar_url in config")
        return SidecarEmbeddingClient(url=config.sidecar_url)
    elif config.strategy == "mock":
        return MockEmbeddingClient(dimension=config.dimension)
    else:
        raise ValueError(f"Unknown embedding strategy: {config.strategy}")
