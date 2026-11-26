"""Shared fixtures. Tests use a tiny fake embedder so they run offline in seconds."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


class FakeEmbedder:
    """Bag-of-words hashed into a fixed-size vector: crude, but similar texts score higher."""

    dim = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        # Split on whitespace and trim punctuation so this works for Sinhala as well as English.
        for word in (w.strip(".,?!:;()[]") for w in text.lower().split()):
            if not word:
                continue
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class FakeLLM:
    name = "fake"

    def __init__(self):
        self.calls = []

    def stream(self, system, messages):
        self.calls.append((system, messages))
        yield "The answer "
        yield "is 42 [1]."


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def sample_pdf(tmp_path):
    """A two-page PDF with distinct topics on each page."""
    path = tmp_path / "handbook.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawString(72, 720, "Vacation policy: employees receive 25 days of paid leave per year.")
    c.showPage()
    c.drawString(72, 720, "Security policy: rotate your laptop password every 90 days.")
    c.showPage()
    c.save()
    return path


@pytest.fixture
def sinhala_pdf():
    """Two Sinhala pages: annual leave (page 1) and laptop password (page 2)."""
    return Path(__file__).parent / "fixtures" / "sinhala_sample.pdf"
