# KI-Veritas — Veritas UI Campus Assistant Chatbot (RAG + Gemini + Chroma)

KI-Veritas is a **campus assistant chatbot** designed to help users quickly find information about **Universitas Indonesia (UI)** using a **Retrieval-Augmented Generation (RAG)** workflow.

The system:
1) retrieves relevant passages from a **PDF knowledge base** stored in a **Chroma** vector database, then  
2) uses **Gemini** to generate an answer **only from the retrieved context**.

> **Guardrail:** If the required information is not present in the knowledge base, the chatbot will respond with **“Tidak ditemukan di dataset.”**

---

## Features
- **RAG pipeline:** Chroma retrieval → Gemini generation
- **Multi-domain vector store:** separated collections for:
  - `umum` (general)
  - `event`
  - `fasilitas` (facilities)
  - `dosen` (lecturers)
- **Multilingual embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Web UI (Flask):** multi chat-session support (new chat, rename, delete) + clear chat
- **Simple guardrails:** model is instructed to answer based on context; otherwise respond “Tidak ditemukan di dataset.”

---

## Tech Stack
- **Backend:** Python + Flask
- **LLM:** Google Gemini (via `google-generativeai`)
- **Vector DB:** Chroma (`chromadb`)
- **Embeddings:** HuggingFace Sentence Transformers (LangChain embedding wrapper)
- **PDF ingestion:** LangChain `PyPDFLoader` + recursive text splitter

---

## Results / Behavior Notes
- Answers are generated **only from retrieved context** (best-effort).
- Retrieval defaults to **top-k = 4** chunks per query.
- Chunking config in `data.py`:
  - `chunk_size = 1500`
  - `chunk_overlap = 200`

---

## Repository Structure

```text
KI-Veritas/
├─ app.py                      # Flask web app + RAG runtime (Gemini)
├─ data.py                     # Indexing pipeline (PDF → Chroma)
├─ get_embedding_function.py   # Embedding function (Sentence Transformers)
├─ requirements.txt
├─ chroma/                     # Persisted vector DB (created after indexing)
│  ├─ umum/
│  ├─ event/
│  ├─ fasilitas/
│  └─ dosen/
└─ data/                       # Put your knowledge-base PDFs here (by domain)
   ├─ umum/
   ├─ event/
   ├─ fasilitas/
   └─ dosen/
```

> If your repo already contains `chroma/` directories, you can run the app immediately (skip indexing).
> If you want to re-index from scratch, delete the subfolders in `chroma/` and run the indexing step again.

---

## Setup

### 1) Create & activate a virtual environment
**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Set Gemini API key (required)
Set the environment variable **`GEMINI_API_KEY`**.

**Windows (PowerShell)**
```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

**macOS / Linux**
```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

Optional: choose model name (default is `gemini-2.5-flash`)
```bash
export GEMINI_MODEL="gemini-2.5-flash"
```

---

## Add Knowledge Base PDFs

Place UI-related PDFs into the `data/` folder **by domain**:

```text
data/
├─ umum/       # general campus info
├─ event/      # events, schedules, announcements
├─ fasilitas/  # facilities, services, locations
└─ dosen/      # lecturer info
```

Example:
```text
data/umum/Peraturan_Akademik_UI.pdf
data/fasilitas/Daftar_Fasilitas_UI.pdf
data/dosen/Profil_Dosen_FMIPA.pdf
```

---

## Build the Vector Database (Indexing)

`data.py` supports indexing **one domain** at a time, or **all domains**.

### Index all domains
```bash
python data.py --domain all
```

### Index a single domain
```bash
python data.py --domain umum
python data.py --domain event
python data.py --domain fasilitas
python data.py --domain dosen
```

What happens:
- PDFs are loaded from `data/<domain>/*.pdf`
- Text is split into chunks
- Chunks are embedded and stored to `chroma/<domain>/`

---

## Run the Web App

```bash
python app.py
```

Then open:
- `http://127.0.0.1:5000`

---

## How It Works (High-Level)

1) **Domain selection**  
   The UI sends a request with a selected domain (`umum/event/fasilitas/dosen`).

2) **Retrieval**  
   The backend queries the corresponding Chroma DB folder:
   - `chroma/umum`, `chroma/event`, `chroma/fasilitas`, `chroma/dosen`

3) **Context + Prompting**  
   The retrieved chunks become “context” passed into Gemini with a strict instruction:
   - answer **only** from context
   - if not found → “Tidak ditemukan di dataset.”

4) **Response returned to UI**  
   The UI displays the answer and keeps chat history per session.

---

## Troubleshooting

### A) `GEMINI_API_KEY not set`
Make sure you exported the env var in the same terminal session where you run `python app.py`.

### B) The chatbot always says “Tidak ditemukan di dataset.”
Common causes:
- `data/<domain>/` folder is empty
- you indexed a different domain than the one selected in UI
- you haven’t run indexing (`python data.py --domain ...`) and `chroma/<domain>/` is missing

### C) Re-indexing / resetting Chroma
Delete the existing persisted folders:
```bash
rm -rf chroma/umum chroma/event chroma/fasilitas chroma/dosen
```
Then re-run indexing.

### D) Dependency errors on install
Try upgrading pip:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Notes & Limitations
- This is a lightweight demo app (single-process Flask).  
- “Answer-from-context-only” is best-effort: always validate critical info against the original PDF source.  
- Large PDF sets will increase indexing time and Chroma folder size.
