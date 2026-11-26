"""Split page text into overlapping chunks small enough to embed well.

This is a minimal recursive splitter: it tries to break on paragraph breaks,
then lines, then sentences, then words, so chunks end at natural boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.loader import Page

SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page_number: int


def chunk_pages(pages: list[Page], chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for page in pages:
        for i, text in enumerate(split_text(page.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{page.source}::p{page.page_number}::c{i}",
                    text=text,
                    source=page.source,
                    page_number=page.page_number,
                )
            )
    return chunks


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    pieces = _split_recursive(text, chunk_size, SEPARATORS)
    return _merge_with_overlap(pieces, chunk_size, overlap)


def _split_recursive(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[str]:
    """Break text into pieces no longer than chunk_size, preferring big separators."""
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        # No separator left: hard cut.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split_recursive(text, chunk_size, rest)

    parts = text.split(sep)
    pieces: list[str] = []
    for j, part in enumerate(parts):
        part = part + (sep if j < len(parts) - 1 else "")
        if not part.strip():
            continue
        pieces.extend(_split_recursive(part, chunk_size, rest))
    return pieces


def _merge_with_overlap(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack pieces into chunks, carrying `overlap` chars of context forward."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            # Start the tail at a word boundary so chunks don't begin mid-word.
            if " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            current = tail if len(tail) + len(piece) <= chunk_size else ""
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks
