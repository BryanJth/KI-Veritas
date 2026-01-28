# KI-Veritas — Veritas UI Campus Assistant Chatbot (RAG + Gemini + Chroma)

KI-Veritas is a campus assistant chatbot designed to help students quickly and clearly understand information related to Universitas Indonesia (UI).  
The system uses a **Retrieval-Augmented Generation (RAG)** approach: answers are generated from context retrieved from a knowledge base (PDF documents) stored in a vector database, then summarized by an LLM.

---

## Feature Summary

- **RAG pipeline:** retrieval (Chroma) → generation (Gemini)
- **Vector database:** Chroma (persisted in the `chroma/` folder)
- **Multilingual embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Domain separation in the vector store:**
  - `general`, `events`, `facilities`, `lecturers`
- **Guardrail:** if the requested information is not available in the knowledge base, the chatbot will respond with *"not found in the dataset"*
- **Web UI:** multi chat-session support (new chat, rename, delete) + clear chat

---

## Project Structure

```text
KI-Veritas/
├─ app.py                  # Web app (Flask) + RAG runtime (Gemini)
├─ data.py                 # Indexing pipeline (PDF → Chroma)
├─ get_embedding_function.py# Embedding function (Sentence-Transformers)
├─ requirements.txt
├─ chroma/                 # Persisted vector DB (indexing outputs)
│  ├─ general/
│  ├─ events/
│  ├─ facilities/
│  └─ lecturers/
└─ data/
   └─ DATASET_KA-I.pdf
