"""
EcodiaOS — External Service Clients

Connection management for Neo4j, TimescaleDB, Redis, LLM, Embedding,
context compression (Stage 1C), and CDP Wallet (Phase 2 Metabolic Layer).
"""

from ecodiaos.clients.context_compression import CompressionMetrics, ContextCompressor
from ecodiaos.clients.embedding import (
    EmbeddingClient,
    VoyageEmbeddingClient,
    cosine_similarity,
    create_embedding_client,
    create_voyage_client,
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
from ecodiaos.clients.wallet import WalletClient

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
    "WalletClient",
]
