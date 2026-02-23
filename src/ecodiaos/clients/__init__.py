"""
EcodiaOS — External Service Clients

Connection management for Neo4j, TimescaleDB, Redis, LLM, and Embedding.
"""

from ecodiaos.clients.embedding import EmbeddingClient, create_embedding_client
from ecodiaos.clients.llm import LLMProvider, create_llm_provider
from ecodiaos.clients.neo4j import Neo4jClient
from ecodiaos.clients.redis import RedisClient
from ecodiaos.clients.timescaledb import TimescaleDBClient

__all__ = [
    "Neo4jClient",
    "TimescaleDBClient",
    "RedisClient",
    "LLMProvider",
    "create_llm_provider",
    "EmbeddingClient",
    "create_embedding_client",
]
