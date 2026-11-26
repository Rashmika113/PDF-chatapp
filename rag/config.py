"""Central configuration, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # Chunking
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))

    # Retrieval
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))

    # Embeddings run locally on CPU, so ingestion is free.
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    # Where Chroma stores vectors. Empty string = in-memory only.
    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", ""))

    # Default LLM provider shown in the UI.
    default_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))


settings = Settings()
