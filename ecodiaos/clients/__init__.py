"""
EcodiaOS — External Service Clients

Connection management for Neo4j, TimescaleDB, Redis, LLM, Embedding,
and context compression (Stage 1C).
"""

from ecodiaos.clients.context_compression import ContextCompressor, CompressionMetrics
from ecodiaos.clients.embedding import (
    EmbeddingClient,
    VoyageEmbeddingClient,
    create_embedding_client,
    create_voyage_client,
    cosine_similarity,
)
from ecodiaos.clients.llm import (
    ExtendedThinkingProvider,
    LLMProvider,
    create_llm_provider,
    create_thinking_provider,
)
from ecodiaos.clients.neo4j import Neo4jClient
from ecodiaos.clients.redis import RedisClient
from ecodiaos.clients.timescaledb import TimescaleDBClient

__all__ = [
    "Neo4jClient",
    "TimescaleDBClient",
    "RedisClient",
    "LLMProvider",
    "ExtendedThinkingProvider",
    "create_llm_provider",
    "create_thinking_provider",
    "EmbeddingClient",
    "VoyageEmbeddingClient",
    "create_embedding_client",
    "create_voyage_client",
    "cosine_similarity",
    "ContextCompressor",
    "CompressionMetrics",
]
