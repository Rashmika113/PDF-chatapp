"""Store and search chunk embeddings with ChromaDB."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import chromadb

from rag.chunker import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity, 1.0 = identical


class VectorStore:
    def __init__(self, persist_dir: str = "", collection: str | None = None):
        self._client = (
            chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        )
        # A unique name per session keeps in-memory stores isolated from each other.
        name = collection or f"docs-{uuid.uuid4().hex[:8]}"
        self._collection = self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source, "page": c.page_number} for c in chunks],
        )

    def search(self, query_embedding: list[float], k: int = 4) -> list[SearchResult]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            chunk = Chunk(id=chunk_id, text=text, source=meta["source"], page_number=int(meta["page"]))
            hits.append(SearchResult(chunk=chunk, score=1.0 - distance))
        return hits

    def sources(self) -> list[str]:
        metas = self._collection.get(include=["metadatas"])["metadatas"] or []
        return sorted({m["source"] for m in metas})

    def count(self) -> int:
        return self._collection.count()
