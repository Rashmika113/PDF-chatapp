"""Local, free text embeddings with sentence-transformers.

The default model, intfloat/multilingual-e5-base, covers 100 languages including
Sinhala and English, and matches across them: an English question can find a
Sinhala passage and the other way round.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def prefixes_for(model_name: str) -> tuple[str, str]:
    """Return the (query, passage) prefixes a model family was trained with.

    Some embedding models expect a short marker in front of the text. Leaving it out
    still works but noticeably hurts retrieval quality.
    """
    name = model_name.lower()
    if "e5" in name:
        return "query: ", "passage: "
    if "bge" in name and "-en" in name:
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""  # e.g. BAAI/bge-m3 and most other models need no prefix


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. The first run downloads it (~1 GB for the default)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.query_prefix, self.passage_prefix = prefixes_for(model_name)
        self._model = _load_model(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [self.passage_prefix + t for t in texts]
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
