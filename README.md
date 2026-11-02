# Veritas V2 — Universitas Indonesia RAG Assistant

A source-aware, Indonesian-language Retrieval-Augmented Generation chatbot for Universitas Indonesia, built on **Flask + Google Gemini + multilingual embeddings + ChromaDB**. This is an incremental improvement of the original university course project—not a separate from-scratch application.

> **Status:** Code-level implementation and offline tests are complete. Live retrieval/answer quality and Gemini integration have **not** yet been benchmarked with an API key. The bundled evaluation questions are unverified examples, not results. Veritas is a demonstration application, not an official university information service.

## Improvements in V2

- **Correctness:** fixes the faculty/program list intent bug and removes the old `0.0` score fallback; handles retrieval failure by refusing unsupported answers instead of inventing confidence
- **Grounding:** configurable maximum Chroma distance, deduplication, prioritized domain retrieval, and empty-context refusal
- **Verifiable sources:** new PDF ingestion records the real file, 1-indexed page number, domain, and stable chunk ID. Supporting PDF references appear under assistant messages when available. The original databases lack page metadata and therefore cannot provide page citations until re-indexed
- **Incremental PDFs:** multiple PDFs supported in `data/` and `data/<domain>/`; re-running the indexer does not duplicate unchanged chunks. Removed chunks from a re-indexed file are pruned. The original `chroma/` folder is preserved
- **Chat UI:** retains the original Flask experience (new/switch/rename/clear/delete chats), adds source panels and a processing indicator; server-side filesystem sessions replace client-side cookie history
- **Model SDK:** uses Google's `google-genai` SDK rather than the older `google-generativeai`
- **Evaluation:** scripts for retrieval ranking metrics and manually labeled answer quality; no simulated benchmark outcomes
- **Testing:** offline pytest suite for routing, distance filtering, citations, ingestion provenance, deterministic IDs, incremental updates, and evaluation integrity

## Architecture

```text
  User / browser
       |
   Flask app.py  <----> filesystem-backed chat sessions
       |
   Intent + follow-up context (veritas_core.py)
       |
   Ranked, distance-filtered Chroma retrieval
       |        
       +----> chroma_verified_2026/{umum,event,fasilitas,dosen}
       |             ^
       |             | data.py: PDF pages -> chunks + metadata + IDs
       |             | data/**/*.pdf
       |
    Supported context [Dokumen N] + short chat history
       |
    Gemini (google-genai)
       |
    Grounded answer + retrieved source/page list
       |
    Flask chat UI

 eval_rag.py      -> Recall@K / Precision@K / MRR / retrieval latency
 eval_answers.py  -> human-labelled accuracy / groundedness / refusal metrics
```

## Requirements

- Python **3.10+** (3.11 recommended)
- An API key from Google AI Studio for actual chatbot responses
- Enough RAM/disk for the Sentence Transformers model and its PyTorch dependency
- Internet on first setup to install packages and download the embedding model

## Local setup (Windows PowerShell)

Open a terminal in the **Veritas-V2** directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and replace the placeholder `GEMINI_API_KEY` with your own key; change `SECRET_KEY` to a long random secret. `python-dotenv` loads `.env` when `app.py` starts. **Never commit `.env`.**

### Recommended: re-index for source citations

```powershell
python data.py --domain all
```

The original PDF is archived as `archive_legacy_original/DATASET KA-I.pdf` and is **not** indexed by default. Verified PDFs are in `data/<domain>/`. You can also add documents such as `data/umum/Aturan_Akademik.pdf`, `data/dosen/Profil_Dosen.pdf`, or `data/event/Kegiatan.pdf`.

New indexes are created in `chroma_verified_2026/` by default. The indexer reads `.env` just like the Flask app, so `CHROMA_PATH` stays consistent. **Do not delete `chroma/`**: that is the original project's database, and its historical lecturer-profile lookup is still used by certain legacy handlers. Repeated indexing of an unchanged PDF skips existing chunk IDs; changes to a PDF replace stale chunks from that file only. For documents removed entirely from disk, stale database entries must still be removed manually.

### Run

```powershell
python app.py
```

Visit **http://127.0.0.1:5001**. Run `python data.py` before launching the app to receive page-aware citations; restart Flask after re-indexing so Chroma connections reload.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your own API key + secret
python data.py --domain all
python app.py
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | empty | Google Gemini API key; accepts `GOOGLE_API_KEY` as a fallback |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model ID (change if unavailable) |
| `SECRET_KEY` | development placeholder | Set a secure random secret before deployment |
| `CHROMA_PATH` | `chroma_verified_2026` | Path for the verified index; with `VERIFIED_ONLY=1` there is no legacy fallback |
| `MAX_DISTANCE` | `1.2` | Max Chroma **distance** accepted (smaller = closer) |
| `RETRIEVAL_TOP_K` | `5` | Max chunks sent to Gemini |
| `COOKIE_SECURE` | `0` | Set `1` when served over HTTPS |
| `TRUST_PROXY` | `0` | Set `1` **only** behind a trusted reverse proxy |
| `PORT` | `5001` | Local Flask port |
| `SESSION_FILE_DIR` | `instance/sessions` | Filesystem session storage (ignored by git) |

**Threshold caution:** `MAX_DISTANCE=1.2` is an unvalidated initial setting for normalized MiniLM embeddings in an L2 Chroma collection. It is **not a measured optimal cutoff**. Tune it using human-verified questions and inspect false refusals / false inclusions. Changing the embedding model requires building new indexes. The legacy Chroma collections' distance semantics should also be checked against the stored embedding configuration.

For offline baseline comparison using the original stores, set `CHROMA_PATH=chroma` in your environment, restart the process, and use `python eval_rag.py --run-retrieval`. The script does not automatically compare old and new runs; keep and compare their separate output files. Original indexes generally lack `chunk_id` labels, limiting quantitative comparisons until human-verified source identifiers are aligned.

## Evaluating performance

```powershell
python -m pytest -q tests
python eval_rag.py
python eval_answers.py
```

The default scripts validate the evaluation schema and write reports, but **will not output accuracy claims** for the unverified templates. For actual retrieval evaluation:

1. Run the V2 indexer to generate stable chunk IDs
2. Manually annotate real Indonesian questions with expected answers and ground-truth relevant `chunk_id` values, following [`evaluation/ANNOTATION_GUIDE.md`](evaluation/ANNOTATION_GUIDE.md)
3. Mark only reviewed entries as `human_verified`
4. Run:

```powershell
python eval_rag.py --dataset evaluation/your_verified_questions.jsonl --run-retrieval --output evaluation/verified_results.json
python eval_answers.py --annotations evaluation/your_verified_answers.jsonl --output evaluation/verified_answer_results.json
```

Retrieval metrics: Recall@K, Precision@K, MRR, and per-query retrieval latency. For unanswerable queries, assess correct refusal separately rather than inventing relevant documents. Answer groundedness/correctness metrics require humans to review actual generated answers (no automatic LLM-as-judge in this release).

## Known limitations / recommended next improvements

- **No production performance claim** without a labeled benchmark. Both live Gemini and complete Flask integration still need to be smoke-tested in your local environment with dependencies and an API key.
- PDF **image-only/scanned** pages are skipped; no OCR is included.
- Root-level mixed-topic PDFs are tagged using straightforward page-level keywords. Explicit `data/<domain>/` folders give more consistent domain labels. The original 22-page combined dataset may include different subjects on one page.
- Older lecturer lookup/list logic uses legacy person-profile metadata. Newly ingested generic lecturer PDFs use paragraph chunks and do not yet populate that specialized profile index; regular RAG queries still work.
- Source panels show the **retrieved supporting pages**, not a proof that every generated assertion is entailed by those pages. Verify cited claims before publishing academic/administrative guidance.
- Follow-up contextualization uses simple text heuristics; pronouns and long discussions may be ambiguous.
- File-backed chat sessions are suitable for local/small deployments. For production or multi-instance deployment, replace with Redis/database storage and configure HTTPS, trusted proxies, and secret management.
- Consider testing BM25 + vector fusion, a cross-encoder reranker, and better model/chunk-size choices **only after collecting a benchmark**.

## Repository map

```text
app.py                      Original Flask UI + V2 integration
veritas_core.py             RAG retrieval/guardrails/intent/follow-up helpers
data.py                     Source-aware, incremental PDF indexer
legacy_ingestion.py         Archived original extraction helpers (do not execute)
get_embedding_function.py   Original multilingual MiniLM embeddings
chroma/                     Original legacy indexes (preserved)
archive_legacy_original/    Original course PDF (not indexed by default)
chroma_verified_2026/      Verified Chroma indexes (generated, git-ignored)
eval_rag.py                 Offline/online retrieval evaluation
eval_answers.py             Report on human-annotated model responses
evaluation/                 Templates + annotation guide
tests/                      Unit/regression/ingestion tests
.env.example                Local environment settings template
```

This code preserves the Veritas course-project architecture while adding safer retrieval and measurable evaluation infrastructure. It should be presented in a portfolio as **a V2 engineering improvement**, not as a proven accuracy gain until benchmark evidence exists.


## Verified knowledge base add-on (1 November 2026)

This package ships 104 curated UI QA entries from 51 official `*.ui.ac.id` sources, separated into `data/umum`, `data/fasilitas`, `data/event`, and `data/dosen`. The original `DATASET KA-I.pdf` has been **archived** in `archive_legacy_original/` and is **not indexed** by the new command, due to outdated admission and SIAK-NG information and other uncited claims. Original Chroma files remain untouched in `chroma/` but are **not read** in the default `VERIFIED_ONLY=1` mode. In this mode direct legacy faculty/lecturer shortcuts are intentionally bypassed, using evidence-backed PDF RAG and an explicitly partial, reviewed mathematics lecturer directory.

For this distribution, use:

```powershell
python data.py --domain all --output-dir chroma_verified_2026
python app.py
```

The default `.env.example` has `CHROMA_PATH=chroma_verified_2026` and `VERIFIED_ONLY=1`. Copy `.env.example` to `.env` and enter your own Gemini API key. **Run indexing before launching**. Legacy-only shortcuts are still available by explicitly setting `VERIFIED_ONLY=0`, but that mode can use old unverified content and is not recommended.

The PDFs include source URLs and the source register is `evaluation/official_source_register.csv`. The full machine-readable corpus is `evaluation/official_qa_verified_2026.jsonl`. Dates, tuition, hours and personnel status still need periodic revalidation.


## Fix: very broad lecturer questions (November 2026)

List-style requests such as `siapa aja dosen UI?` no longer retrieve a random single profile:

- Broad requests ask which faculty or department the user means; no Gemini call is needed
- `Matematika` as a follow-up shows **six selected documented names**, clearly labeled as an **incomplete list**; it is not a roster of all Mathematics lecturers
- Questions about a named lecturer or a research specialty continue through normal RAG
- Repeating a short self-contained question does not automatically append the preceding question
- The verified directory source file is `data/dosen/lecturer_directory_verified.json`; each name links to a staff handbook included in the audited knowledge base
- If no verified Chroma index exists, Veritas tells you to run the indexer instead of silently falling back

**Important:** The newly installed Flask app won't automatically update a Flask server already running.
Stop the old server (`Ctrl+C`), install from the updated project directory, run `python data.py --domain all`, and then restart `python app.py`. Keep `VERIFIED_ONLY=1` and `CHROMA_PATH=chroma_verified_2026` in `.env`. The dataset does not contain all UI lecturers, so the assistant should not claim to list everyone.

To verify your setup without printing secrets, run `python check_kb.py`.
