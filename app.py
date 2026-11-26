"""Streamlit UI: upload PDFs in the sidebar, chat with them in the main pane.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from rag import RAGPipeline
from rag.config import settings
from rag.llm import PROVIDERS, get_llm

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def _warm_embedder():
    from rag.embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(settings.embedding_model)


def get_pipeline() -> RAGPipeline:
    # One pipeline (and vector store) per browser session; the model is shared.
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline(embedder=_warm_embedder())
        st.session_state.ingested = set()
        st.session_state.messages = []
    return st.session_state.pipeline


pipeline = get_pipeline()

# ---- Sidebar: model settings and uploads --------------------------------
with st.sidebar:
    st.header("⚙️ Model")
    provider_keys = list(PROVIDERS)
    provider = st.selectbox(
        "LLM provider",
        provider_keys,
        index=provider_keys.index(settings.default_provider) if settings.default_provider in PROVIDERS else 0,
        format_func=lambda k: PROVIDERS[k].label,
    )
    info = PROVIDERS[provider]
    model = st.text_input("Model", value=info.default_model)
    api_key = None
    if info.needs_api_key:
        api_key = st.text_input(
            f"API key ({info.api_key_env})", type="password", help="Leave empty to use the .env value."
        )
    if provider == "ollama":
        st.caption(f"Make sure Ollama is running and you've pulled the model: `ollama pull {model}`")

    st.header("📄 Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    for upload in uploads or []:
        file_id = f"{upload.name}:{upload.size}"
        if file_id in st.session_state.ingested:
            continue
        with st.spinner(f"Indexing {upload.name}..."):
            n_chunks = pipeline.ingest(upload, source=upload.name)
        st.session_state.ingested.add(file_id)
        if n_chunks:
            st.success(f"{upload.name}: {n_chunks} chunks indexed")
        else:
            st.warning(f"{upload.name}: no text found (is it a scanned PDF?)")

    if pipeline.store.count():
        st.caption(f"{pipeline.store.count()} chunks from: " + ", ".join(pipeline.store.sources()))

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---- Main pane: chat ----------------------------------------------------
st.title("📄 Chat with your PDFs")
st.caption("Upload documents, ask questions, and get answers with page-level citations.")


def show_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s['source']}, page {s['page']}** · similarity {s['score']:.2f}")
            st.caption(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        try:
            llm = get_llm(provider, model=model, api_key=api_key)
            result = pipeline.answer(question, llm, history=history)
            text = st.write_stream(result.tokens)
        except Exception as exc:  # show provider errors in the chat instead of crashing
            text = f"⚠️ {type(exc).__name__}: {exc}"
            st.error(text)
            result = None

        sources = [
            {"source": s.chunk.source, "page": s.chunk.page_number, "score": s.score, "text": s.chunk.text}
            for s in (result.sources if result else [])
        ]
        if sources:
            show_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": text, "sources": sources})
