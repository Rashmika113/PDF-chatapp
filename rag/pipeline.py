"""The RAG pipeline: ingest PDFs, retrieve relevant chunks, and answer with citations.

upload -> extract text -> chunk -> embed -> store      (ingest)
question -> embed -> top-k search -> prompt LLM        (answer)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import BinaryIO

from rag.chunker import chunk_pages
from rag.config import Settings
from rag.config import settings as default_settings
from rag.embeddings import Embedder, SentenceTransformerEmbedder
from rag.llm import LLM, Message
from rag.loader import load_pdf
from rag.vector_store import SearchResult, VectorStore

SYSTEM_PROMPT = """You answer questions using only the document excerpts provided.

Rules:
- Base every statement on the excerpts.
- If the excerpts don't contain the answer, say you couldn't find it in the documents.
- Cite the excerpts you used inline with their tag, for example [1] or [2][3].
- Be concise. Use short paragraphs or bullet points when it helps."""

# How many earlier chat turns to send, so follow-up questions have context.
HISTORY_TURNS = 6


@dataclass
class Answer:
    """A streamed answer. Iterate `tokens` for text; `sources` holds what was retrieved."""

    tokens: Iterator[str]
    sources: list[SearchResult] = field(default_factory=list)


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        config: Settings = default_settings,
    ):
        self.config = config
        self.embedder = embedder or SentenceTransformerEmbedder(config.embedding_model)
        self.store = store or VectorStore(config.chroma_dir)

    # ---- ingest -------------------------------------------------------
    def ingest(self, file: str | BinaryIO, source: str | None = None) -> int:
        """Add one PDF to the index. Returns the number of chunks stored."""
        pages = load_pdf(file, source)
        chunks = chunk_pages(pages, self.config.chunk_size, self.config.chunk_overlap)
        if chunks:
            self.store.add(chunks, self.embedder.embed_documents([c.text for c in chunks]))
        return len(chunks)

    # ---- retrieve -----------------------------------------------------
    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        return self.store.search(self.embedder.embed_query(question), k or self.config.top_k)

    # ---- answer -------------------------------------------------------
    def answer(self, question: str, llm: LLM, history: list[Message] | None = None) -> Answer:
        sources = self.retrieve(question)
        if not sources:
            return Answer(tokens=iter(["Upload a PDF first, then ask me about it."]), sources=[])

        messages = [
            *(history or [])[-HISTORY_TURNS * 2 :],
            {"role": "user", "content": build_prompt(question, sources)},
        ]
        return Answer(tokens=llm.stream(SYSTEM_PROMPT, messages), sources=sources)


def build_prompt(question: str, sources: list[SearchResult]) -> str:
    excerpts = "\n\n".join(
        f"[{i}] (from {s.chunk.source}, page {s.chunk.page_number})\n{s.chunk.text}"
        for i, s in enumerate(sources, start=1)
    )
    return f"<excerpts>\n{excerpts}\n</excerpts>\n\nQuestion: {question}"
