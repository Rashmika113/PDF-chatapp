import pytest

from rag.chunker import chunk_pages, split_text
from rag.loader import Page


def test_short_text_is_one_chunk():
    assert split_text("hello world", chunk_size=100, overlap=10) == ["hello world"]


def test_chunks_respect_size_limit():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = split_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_consecutive_chunks_overlap():
    text = ". ".join(f"Sentence number {i} is here" for i in range(100))
    chunks = split_text(text, chunk_size=150, overlap=50)
    for a, b in zip(chunks, chunks[1:], strict=False):
        # The next chunk starts with text carried over from the end of the previous one.
        assert b.split()[0] in a


def test_no_text_is_lost():
    words = [f"w{i}" for i in range(500)]
    chunks = split_text(" ".join(words), chunk_size=100, overlap=0)
    assert " ".join(chunks).split() == words


def test_chunk_pages_keeps_page_metadata():
    pages = [Page("a.pdf", 1, "first page"), Page("a.pdf", 2, "second page")]
    chunks = chunk_pages(pages, chunk_size=100, overlap=10)
    assert [(c.source, c.page_number) for c in chunks] == [("a.pdf", 1), ("a.pdf", 2)]
    assert len({c.id for c in chunks}) == 2


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_pages([Page("a.pdf", 1, "x")], chunk_size=10, overlap=10)
