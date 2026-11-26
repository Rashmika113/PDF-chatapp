"""Local, free text embeddings with sentence-transformers."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. The first run downloads it (~130 MB)."""

    # BGE models retrieve better when queries carry this instruction prefix.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = _load_model(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        prefix = self.QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self._model.encode(prefix + text, normalize_embeddings=True).tolist()


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
