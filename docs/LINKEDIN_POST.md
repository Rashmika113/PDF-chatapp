# LinkedIn post draft

Replace the bracketed parts, attach the demo GIF (LinkedIn autoplays GIFs and video better than links), and put the repo link in the first comment if you want more reach.

---

I built a "Chat with your PDFs" app from scratch to really understand how Retrieval-Augmented Generation (RAG) works. 📄🤖

You upload documents, ask questions in English or Sinhala (සිංහල), and get answers with page-level citations, so you can check every claim against the source.

Instead of reaching for a framework, I wrote each step myself so I could see what's going on:

🔹 Extract text page by page with pypdf
🔹 Split it into overlapping chunks so ideas aren't cut in half
🔹 Embed chunks locally with an open-source multilingual model (multilingual-e5), so indexing is free and Sinhala works too
🔹 Store and search vectors with ChromaDB
🔹 Send the top matches to an LLM with instructions to answer only from the excerpts and cite them

A few things I learned:
• Chunk size and overlap matter more than I expected. Too small loses context, too big dilutes the match.
• Multilingual embeddings match meaning across languages: a Sinhala question can find the right passage in an English PDF.
• Telling the model to say "I couldn't find that" when the excerpts don't cover a question cuts down on made-up answers a lot.
• [Add one surprise from your own testing, e.g. a question it got wrong and why.]

It runs 100% free and local with Ollama, or you can switch to Groq, OpenAI or Claude from the sidebar. Built with Python, Streamlit and pytest, with CI on GitHub Actions.

Next I want to add hybrid search with a reranker and a small evaluation set to measure answer quality.

Code is on GitHub: https://github.com/Rashmika113/PDF-chatapp

#AI #MachineLearning #RAG #LLM #Python #GenerativeAI #NLP #SriLanka #OpenToWork
