# KI-Veritas — Veritas Chatbot Asisten Kampus UI (RAG + Gemini + Chroma)

KI-Veritas adalah chatbot asisten kampus untuk membantu mahasiswa memahami informasi seputar Universitas Indonesia (UI) secara ringkas dan terarah. Sistem menggunakan pendekatan **Retrieval-Augmented Generation (RAG)**: jawaban dibangun dari **konteks** yang diambil dari knowledge base (dokumen PDF) melalui vector database, lalu dirangkum oleh LLM.

## Ringkasan Fitur
- **RAG pipeline**: retrieval (Chroma) → generation (Gemini)
- **Vector database**: Chroma (persist di folder `chroma/`)
- **Embedding multilingual**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Domain separation** di vector store:
  - `umum`, `event`, `fasilitas`, `dosen`
- **Guardrail**: jika informasi tidak tersedia pada knowledge base, chatbot menyatakan *tidak ditemukan di dataset*
- **Web UI**: multi chat-session (new chat, rename, delete) + clear chat

## Struktur Proyek
```text
KI-Veritas/
├─ app.py                    # Web app (Flask) + runtime RAG (Gemini)
├─ data.py                   # Indexing pipeline (PDF -> Chroma)
├─ get_embedding_function.py # Embedding function (Sentence-Transformers)
├─ requirements.txt
├─ chroma/                   # Persist vector DB (hasil indexing)
│  ├─ umum/
│  ├─ event/
│  ├─ fasilitas/
│  └─ dosen/
└─ data/
   └─ DATASET KA-I.pdf
```
