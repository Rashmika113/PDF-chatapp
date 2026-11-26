import pytest

from rag.config import Settings
from rag.llm import get_llm
from rag.pipeline import RAGPipeline, build_prompt
from rag.vector_store import VectorStore


@pytest.fixture
def pipeline(fake_embedder):
    return RAGPipeline(embedder=fake_embedder, store=VectorStore(), config=Settings(top_k=1))


def test_ingest_indexes_chunks(pipeline, sample_pdf):
    assert pipeline.ingest(sample_pdf) == 2
    assert pipeline.store.sources() == ["handbook.pdf"]


def test_ingesting_twice_does_not_duplicate(pipeline, sample_pdf):
    pipeline.ingest(sample_pdf)
    pipeline.ingest(sample_pdf)
    assert pipeline.store.count() == 2


def test_retrieve_finds_the_relevant_page(pipeline, sample_pdf):
    pipeline.ingest(sample_pdf)
    [hit] = pipeline.retrieve("how often should I rotate my password?")
    assert hit.chunk.page_number == 2
    [hit] = pipeline.retrieve("how many days of paid vacation leave?")
    assert hit.chunk.page_number == 1


def test_answer_streams_and_returns_sources(pipeline, sample_pdf, fake_llm):
    pipeline.ingest(sample_pdf)
    result = pipeline.answer("password rotation?", fake_llm, history=[{"role": "user", "content": "hi"}])
    assert "".join(result.tokens) == "The answer is 42 [1]."
    assert result.sources[0].chunk.page_number == 2

    system, messages = fake_llm.calls[0]
    assert "only the document excerpts" in system
    assert messages[0] == {"role": "user", "content": "hi"}
    assert "page 2" in messages[-1]["content"]


def test_answer_without_documents_asks_for_upload(pipeline, fake_llm):
    result = pipeline.answer("anything?", fake_llm)
    assert "Upload a PDF" in "".join(result.tokens)
    assert fake_llm.calls == []


def test_build_prompt_numbers_excerpts(pipeline, sample_pdf):
    pipeline.ingest(sample_pdf)
    prompt = build_prompt("q", pipeline.retrieve("policy", k=2))
    assert "[1] (from handbook.pdf" in prompt and "[2] (from handbook.pdf" in prompt


def test_get_llm_requires_key_for_hosted_providers(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_llm("groq")


def test_get_llm_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_llm("nope")


def test_ollama_needs_no_key():
    assert get_llm("ollama").name == "ollama"
