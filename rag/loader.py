"""Turn PDF files into plain text, one record per page."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    source: str  # file name, used for citations
    page_number: int  # 1-based, as a human would read it
    text: str


def load_pdf(file: str | Path | BinaryIO, source: str | None = None) -> list[Page]:
    """Extract text from every page of a PDF.

    `file` can be a path or any binary file-like object (Streamlit uploads are).
    Pages with no extractable text (e.g. scanned images) are skipped.
    """
    if source is None:
        source = Path(file).name if isinstance(file, (str, Path)) else getattr(file, "name", "document.pdf")

    reader = PdfReader(file)
    pages: list[Page] = []
    for index, pdf_page in enumerate(reader.pages, start=1):
        text = _clean(pdf_page.extract_text() or "")
        if text:
            pages.append(Page(source=source, page_number=index, text=text))
    return pages


def _clean(text: str) -> str:
    """Normalise whitespace that PDF extraction tends to mangle."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()
