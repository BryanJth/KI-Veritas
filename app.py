from flask import Flask, render_template_string, request, session, redirect, url_for
import re
import uuid
import os
import unicodedata
from datetime import datetime, timezone

# Chroma import (prefer langchain_chroma if available)
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document

from get_embedding_function import get_embedding_function

# Security extras
from flask_wtf.csrf import CSRFProtect, generate_csrf
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

# Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None



# CONFIG
APP_NAME = "Veritas"
APP_TAGLINE = "Chatbot Asisten kampus"
BRAND_IMAGE_URL = "https://i.imgur.com/ud8HfJd.png"
FAVICON_URL = "https://i.imgur.com/ud8HfJd.png"

CHROMA_PATH = "chroma"

MAX_QUESTION_CHARS = 800

RATE_LIMIT_DEFAULT = "20 per minute"
RATE_LIMIT_ASK = "10 per minute"

PROMPT_TEMPLATE = """
Kamu adalah asisten informasi Universitas Indonesia.

Aturan keras:
- Jawab hanya memakai fakta dari KONTEKS (knowledge base), bukan pengetahuan luar.
- Jika KONTEKS tidak memuat jawaban, katakan persis: "Saya tidak menemukan informasi itu di dataset yang tersedia."
- Jawaban harus Bahasa Indonesia, ringkas, nyambung (2–6 kalimat).
- Output harus teks biasa (tanpa Markdown: jangan pakai **, *, #, backticks).

{focus_hint}

Riwayat singkat percakapan:
{history}

KONTEKS:
{context}

Pertanyaan pengguna:
{question}
""".strip()



# Cache objects
_EMBEDDINGS = None
_DB_CACHE = {}
_GEMINI_MODEL = None
_DOSEN_LOOKUP = None
_LAST_CONTEXT = ""
_FAKULTAS_CACHE = None  # {"fakultas": [(idx_or_none, name)], "all": [(idx_or_none, name)]}



# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()


def _get_gemini_model():
    global _GEMINI_MODEL
    if genai is None:
        return None
    if not GEMINI_API_KEY:
        return None
    if _GEMINI_MODEL is None:
        genai.configure(api_key=GEMINI_API_KEY)
        _GEMINI_MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)
    return _GEMINI_MODEL


# DB helpers
CHROMA_DIRS = {
    "umum": os.path.join(CHROMA_PATH, "umum"),
    "event": os.path.join(CHROMA_PATH, "event"),
    "fasilitas": os.path.join(CHROMA_PATH, "fasilitas"),
    "dosen": os.path.join(CHROMA_PATH, "dosen"),
}


def get_db(domain: str = "umum"):
    global _EMBEDDINGS, _DB_CACHE
    if domain not in CHROMA_DIRS:
        domain = "umum"

    if _EMBEDDINGS is None:
        _EMBEDDINGS = get_embedding_function()

    if domain not in _DB_CACHE:
        _DB_CACHE[domain] = Chroma(
            persist_directory=CHROMA_DIRS[domain],
            embedding_function=_EMBEDDINGS
        )
    return _DB_CACHE[domain]


def _available_domains():
    # only include domains that exist on disk (avoid Chroma errors)
    out = []
    for d, p in CHROMA_DIRS.items():
        if os.path.isdir(p):
            out.append(d)
    return out or ["umum"]



# Text utils
_DEGREE_TOKENS = {
    "phd", "ph", "d", "ph.d",
    "msc", "m", "sc", "m.sc",
    "msi", "m.si", "si",
    "dea", "mt", "ssi", "s.si", "s.sos", "st", "s.t",
    "rer", "nat", "dra", "ir", "apt",
    "mkom", "m.kom",
    "prof", "dr",
}


def _norm_basic(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_strict(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_name(name: str) -> str:
    n = _norm_strict(name)
    parts = [p for p in n.split() if p not in _DEGREE_TOKENS]
    return " ".join(parts).strip()


def _title_from_norm(n: str) -> str:
    return " ".join(w.capitalize() for w in (n or "").split()).strip()


def _clean_answer(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = t.replace("**", "").replace("__", "")
    t = re.sub(r"(?m)^\s*[\-\*•]\s+", "", t)
    t = re.sub(r"(?m)^\s*#+\s*", "", t)
    # keep newlines (UI supports pre-wrap), but compress too many spaces
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    return t



# Fakultas extractor (deterministic)
def _extract_enumerated_units_from_text(text: str):
    """
    Parse patterns like:
      1) Fakultas Kedokteran
      2. Fakultas Teknik
    Also works if separated by ';'
    Return list of (idx:int, name:str)
    """
    if not text:
        return []
    # Allow start of line OR after newline OR after semicolon
    pat = re.compile(r"(?:^|[\n;])\s*(\d{1,2})\s*[\)\.\-:]\s*([^\n;]{3,220})")
    out = []
    for idx_s, name in pat.findall(text):
        try:
            idx = int(idx_s)
        except Exception:
            continue
        nm = re.sub(r"\s+", " ", (name or "")).strip()
        nm = nm.strip(" -–—:,.")
        low = (nm or "").lower()

        # Keep only things that look like UI units
        if not (low.startswith("fakultas") or low.startswith("program") or low.startswith("sekolah")):
            continue

        out.append((idx, nm))
    return out


def _extract_fakultas_phrases(text: str):
    """
    Fallback: capture occurrences like 'Fakultas Kedokteran', 'Fakultas Ilmu Komputer', etc.
    """
    if not text:
        return []
    pat = re.compile(r"\bFakultas\s+[A-Z][A-Za-z0-9\-\&\.\s]{2,90}")
    found = []
    for m in pat.findall(text):
        nm = re.sub(r"\s+", " ", m).strip().strip(" -–—:,.")
        if nm:
            found.append(nm)
    return found


def _build_fakultas_cache():
    texts = []
    for d in _available_domains():
        try:
            items = get_db(d).get(include=["documents"])
        except Exception:
            continue
        docs = items.get("documents") or []
        for t in docs:
            if t:
                texts.append(t)

    enum_map = {}  # name_norm -> (idx, name)
    for t in texts:
        for idx, nm in _extract_enumerated_units_from_text(t):
            key = _norm_strict(nm)
            if not key:
                continue
            if key not in enum_map or idx < enum_map[key][0]:
                enum_map[key] = (idx, nm)

    # Also gather faculty phrases (no idx)
    phrase_set = {}
    for t in texts:
        for nm in _extract_fakultas_phrases(t):
            key = _norm_strict(nm)
            if not key:
                continue
            # keep if not already in enum_map
            if key not in enum_map and key not in phrase_set:
                phrase_set[key] = (None, nm)

    # Merge
    all_items = list(enum_map.values()) + list(phrase_set.values())

    # Dedup by normalized name again
    dedup = {}
    for idx, nm in all_items:
        key = _norm_strict(nm)
        if not key:
            continue
        if key not in dedup:
            dedup[key] = (idx, nm)
        else:
            # prefer numbered one
            if dedup[key][0] is None and idx is not None:
                dedup[key] = (idx, nm)

    all_items2 = list(dedup.values())

    def sort_key(x):
        idx, nm = x
        idx_sort = idx if idx is not None else 9999
        return (idx_sort, (nm or "").lower())

    all_items2.sort(key=sort_key)

    fakultas = [x for x in all_items2 if (x[1] or "").lower().startswith("fakultas")]
    return {"fakultas": fakultas, "all": all_items2}


def format_list_fakultas(include_program_sekolah: bool = False) -> str:
    global _FAKULTAS_CACHE
    if _FAKULTAS_CACHE is None:
        _FAKULTAS_CACHE = _build_fakultas_cache()

    items = _FAKULTAS_CACHE["all" if include_program_sekolah else "fakultas"]
    if not items:
        return "Saya tidak menemukan informasi itu di dataset yang tersedia."

    # If enumerated numbering exists, keep it; otherwise re-number sequentially.
    has_numbering = any(idx is not None for idx, _nm in items)
    if has_numbering and not include_program_sekolah:
        items = [(idx, nm) for idx, nm in items if idx is not None]

    lines = []
    if include_program_sekolah:
        lines.append("Daftar fakultas/program/sekolah yang ditemukan di dataset:")
    else:
        lines.append("Daftar fakultas yang ditemukan di dataset:")

    if has_numbering:
        for idx, nm in items:
            if idx is None:
                lines.append(f"- {nm}")
            else:
                lines.append(f"{idx}. {nm}")
    else:
        for i, (_idx, nm) in enumerate(items, start=1):
            lines.append(f"{i}. {nm}")

    return "\n".join(lines)


# Intent detection (non-person)
def detect_intent_non_person(q: str) -> str:
    ql = (q or "").lower()
    if any(x in ql for x in ["wisuda", "pkkmb", "omb", "seminar", "kuliah umum", "career fair", "dies natalis", "lomba", "event", "kegiatan"]):
        return "event"
    if any(x in ql for x in ["perpustakaan", "rsui", "balairung", "sport center", "bis kuning", "asrama", "student center", "danau", "masjid", "atm", "bank", "fasilitas"]):
        return "fasilitas"
    if any(x in ql for x in ["dosen", "prof", "dr.", "siapa "]):
        # not strictly needed; direct dosen handler covers
        return "dosen"
    return "umum"


def is_list_dosen(q: str) -> bool:
    ql = (q or "").lower()
    return any(x in ql for x in ["daftar dosen", "list dosen", "nama dosen", "dosen apa saja"])


def is_list_fakultas(q: str) -> bool:
    qn = _norm_strict(q)
    if "fakultas" not in qn:
        return False
    triggers = [
        "daftar fakultas", "list fakultas", "nama fakultas",
        "sebut fakultas", "sebutkan fakultas", "sebutin fakultas",
        "semua fakultas", "fakultas apa saja", "fakultas apa aja",
        "berikan semua fakultas", "tuliskan fakultas"
    ]
    return any(_norm_strict(t) in qn for t in triggers) or any(t in qn for t in ["daftar", "list", "sebut", "sebutkan", "tuliskan", "berikan"])


def is_list_fakultas_program_sekolah(q: str) -> bool:
    qn = _norm_strict(q)
    if "program" in qn or "sekolah" in qn:
        return True
    # combined phrasing
    return "fakultas program sekolah" in qn or "fakultas program" in qn or "program sekolah" in qn


def _is_pronoun_followup(q: str) -> bool:
    qn = _norm_strict(q)
    return any(x in qn.split() for x in ["dia", "beliau", "itu", "tsb"]) or "dosen itu" in qn or "orang itu" in qn


def _is_generic_followup(q: str) -> bool:
    qn = _norm_basic(q)
    toks = qn.split()
    if len(toks) <= 3:
        return True
    return any(k in qn for k in ["sebut", "sebutkan", "coba", "lanjut", "detail", "jelasin", "jelaskan", "yang mana", "apa aja", "berapa saja", "list", "daftar"])



# History helpers
def build_history_text(messages):
    if not messages:
        return "Tidak ada percakapan sebelumnya."
    lines = []
    for m in messages[-10:]:
        role = "Pengguna" if m.get("role") == "user" else "Asisten"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "Tidak ada percakapan sebelumnya."



# DOSEN lookup (direct match)
def _build_dosen_lookup():
    db = get_db("dosen")
    data = db.get(include=["documents", "metadatas"])
    lookup = {}

    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    for doc, meta in zip(docs, metas):
        if not doc:
            continue
        meta = meta or {}
        raw = (meta.get("nama") or "").strip()
        n1 = (meta.get("nama_norm") or "").strip() or _normalize_name(raw)
        if n1:
            lookup[n1] = doc

        # heuristic: last two tokens of raw name
        toks = [t for t in _norm_strict(raw).split() if t not in _DEGREE_TOKENS]
        if len(toks) >= 2:
            n2 = " ".join(toks[-2:])
            if n2 and n2 not in lookup:
                lookup[n2] = doc

    return lookup


def _try_direct_dosen(question: str):
    global _DOSEN_LOOKUP
    if _DOSEN_LOOKUP is None:
        _DOSEN_LOOKUP = _build_dosen_lookup()

    qn = _norm_strict(question)

    # pattern: "siapa <nama>" / "profil <nama>" / "tentang <nama>"
    m = re.search(r"\b(?:siapa|profil|tentang)\b\s+(.+)$", qn)
    if m:
        tail = m.group(1)
        tail = re.sub(r"\b(dosen|ui|fmipa|universitas)\b", " ", tail)
        tail = re.sub(r"\s+", " ", tail).strip()
        if tail and tail in _DOSEN_LOOKUP:
            return _DOSEN_LOOKUP[tail]

    # fallback: substring match
    for nama_norm, doc in _DOSEN_LOOKUP.items():
        if nama_norm and nama_norm in qn:
            return doc

    return None


def _extract_name_from_dosen_doc(doc_text: str) -> str:
    if not doc_text:
        return ""
    m = re.search(r"Nama:\s*(.+)", doc_text)
    return m.group(1).strip() if m else ""


_BAD_NAME_TOKENS = {
    "gedung", "balairung", "masjid", "perpustakaan", "rsui", "rumah", "sakit",
    "sport", "center", "danau", "asrama", "bis", "bus", "ukhuwa", "islamiyah",
}


def format_list_dosen():
    db = get_db("dosen")
    data = db.get(include=["metadatas"])
    metas = data.get("metadatas") or []

    items = []
    seen = set()

    for m in metas:
        m = m or {}
        if m.get("domain") and m.get("domain") != "dosen":
            continue
        if m.get("type") and m.get("type") != "person":
            continue

        nama = (m.get("nama") or "").strip()
        if not nama:
            continue

        nama_norm = (m.get("nama_norm") or "").strip() or _normalize_name(nama)
        if not nama_norm or nama_norm in seen:
            continue

        toks = set(nama_norm.split())
        if toks & _BAD_NAME_TOKENS:
            continue

        seen.add(nama_norm)

        idx = m.get("idx")
        try:
            idx_int = int(idx) if idx is not None else 9999
        except Exception:
            idx_int = 9999

        display = _title_from_norm(nama_norm) or nama
        items.append((idx_int, display))

    if not items:
        return "Maaf, daftar dosen tidak ditemukan di database. (Coba re-index: python data.py)"

    items.sort(key=lambda x: (x[0], x[1].lower()))

    lines = [f"Daftar dosen ({len(items)} orang):"]
    for idx_int, display in items:
        if idx_int != 9999:
            lines.append(f"{idx_int}. {display}")
        else:
            lines.append(f"- {display}")

    return "\n".join(lines)


# Retrieval (multi-query variants + multi-domain)
def _query_variants(question: str):
    q = (question or "").strip()
    qn = _norm_strict(q)

    variants = [q]

    q2 = re.sub(
        r"\b(siapa|apa|kapan|dimana|di mana|bagaimana|jelaskan|tolong|info|informasi|tentang|profil)\b",
        " ",
        qn,
    )
    q2 = re.sub(r"\s+", " ", q2).strip()
    if q2 and q2 != qn:
        variants.append(q2)

    out, seen = [], set()
    for v in variants:
        key = (v or "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(v)
    return out


def retrieve_context(question_for_retrieval: str, primary_domain: str):
    variants = _query_variants(question_for_retrieval)

    k_primary = 6
    k_fallback = 3

    domain_order = ["dosen", "umum", "event", "fasilitas"]
    domain_order = [d for d in domain_order if d in _available_domains()]

    domains = [primary_domain] + [d for d in domain_order if d != primary_domain]
    collected = []

    for d in domains:
        db = get_db(d)
        k = k_primary if d == primary_domain else k_fallback
        for v in variants:
            try:
                results = db.similarity_search_with_score(v, k=k)
            except Exception:
                docs = db.similarity_search(v, k=k)
                results = [(doc, 0.0) for doc in docs]

            for doc, score in results:
                text = (doc.page_content or "").strip()
                if not text:
                    continue
                collected.append((d, text, float(score)))

    # de-dup
    uniq, seen = [], set()
    for d, text, score in collected:
        key = (d, text[:220])
        if key in seen:
            continue
        seen.add(key)
        uniq.append((d, text, score))

    # smaller score = closer (commonly)
    uniq.sort(key=lambda x: x[2])

    top = []
    primary_added = False
    for d, text, _score in uniq:
        if len(top) >= 10:
            break
        if d == primary_domain:
            primary_added = True
        top.append((d, text))

    if not primary_added:
        for d, text, _score in uniq:
            if d == primary_domain:
                top = [(d, text)] + top
                top = top[:10]
                break

    context_blocks = []
    for d, text in top:
        context_blocks.append(f"=== {d.upper()} ===\n{text}")

    return "\n\n---\n\n".join(context_blocks).strip()



# Core RAG (Gemini)
def query_rag(query_text: str, messages: list, chat_state: dict) -> str:
    global _LAST_CONTEXT
    _LAST_CONTEXT = ""

    q = (query_text or "").strip()
    if not q:
        return ""

    # Deterministic list handlers (NO LLM)
    if is_list_fakultas_program_sekolah(q):
        chat_state["active_person"] = ""
        chat_state["last_topic"] = "fakultas"
        _LAST_CONTEXT = "daftar fakultas/program/sekolah"
        return format_list_fakultas(include_program_sekolah=True)

    if is_list_fakultas(q):
        chat_state["active_person"] = ""
        chat_state["last_topic"] = "fakultas"
        _LAST_CONTEXT = "daftar fakultas"
        return format_list_fakultas(include_program_sekolah=False)

    if is_list_dosen(q):
        chat_state["active_person"] = ""
        chat_state["last_topic"] = "daftar dosen"
        _LAST_CONTEXT = "daftar dosen"
        return format_list_dosen()

    focus_hint = ""
    context = ""

    # Direct dosen match
    direct_doc = _try_direct_dosen(q)
    if direct_doc:
        context = "=== DOSEN ===\n" + direct_doc
        name = _extract_name_from_dosen_doc(direct_doc)
        if name:
            chat_state["active_person"] = name
            chat_state["last_topic"] = name
        _LAST_CONTEXT = context
    else:
        # Follow-up wiring: active person + generic follow-up topic
        retrieval_q = q

        active_person = (chat_state.get("active_person") or "").strip()
        if active_person and _is_pronoun_followup(q):
            retrieval_q = f"{q} {active_person}"
            focus_hint = f"Catatan: pertanyaan ini merujuk pada {active_person}."

        # generic follow-up (e.g., "sebut coba", "lanjut")
        last_topic = (chat_state.get("last_topic") or "").strip()
        if (not active_person) and last_topic and _is_generic_followup(q):
            # If last topic was "fakultas", treat generic follow-up as list request if it contains list verbs
            if last_topic == "fakultas" and any(k in _norm_strict(q) for k in ["sebut", "sebutkan", "tuliskan", "daftar", "list"]):
                chat_state["active_person"] = ""
                _LAST_CONTEXT = "daftar fakultas"
                return format_list_fakultas(include_program_sekolah=False)

            retrieval_q = f"{q} {last_topic}".strip()
            if not focus_hint:
                focus_hint = f"Catatan: pertanyaan ini adalah lanjutan dari topik {last_topic}."

        primary = detect_intent_non_person(q)

        # if question mentions fakultas, store topic to help follow-ups
        if "fakultas" in _norm_strict(q):
            chat_state["last_topic"] = "fakultas"

        context = retrieve_context(retrieval_q, primary)
        _LAST_CONTEXT = context

        # update last_topic (don’t overwrite with generic)
        if not _is_generic_followup(q) and "fakultas" not in _norm_strict(q):
            chat_state["last_topic"] = q

    model = _get_gemini_model()
    if model is None:
        if genai is None:
            return "⚠️ Library Gemini belum terpasang. Install dulu: pip install google-generativeai"
        return "⚠️ GEMINI_API_KEY belum diset. Set environment variable GEMINI_API_KEY sebelum menjalankan."

    prompt = PROMPT_TEMPLATE.format(
        focus_hint=focus_hint,
        history=build_history_text(messages),
        context=context if context else "(kosong)",
        question=q,
    )

    try:
        raw = model.generate_content(prompt).text.strip()
    except Exception:
        return "Terjadi error saat memproses. Silakan coba lagi."

    return _clean_answer(raw)


# Existing helpers
def extract_keywords(query: str):
    q = _norm_basic(query)
    tokens = re.findall(r"[a-z0-9]+", q)
    stop = {
        "siapa", "itu", "di", "yang", "dan", "atau", "pada",
        "dari", "untuk", "apakah", "apa", "bagaimana", "kapan", "dimana"
    }
    tokens = [t for t in tokens if len(t) >= 2 and t not in stop]
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:10]


def _make_title_from_context_or_keywords(q: str, context_text: str) -> str:
    kws = extract_keywords(q)
    ctx = (context_text or "").strip()

    if not ctx:
        return (" ".join(kws[:6]))[:60] if kws else "New chat"

    raw_lines = []
    for block in ctx.split("---"):
        for ln in block.splitlines():
            ln = re.sub(r"\s+", " ", (ln or "").strip())
            if ln:
                raw_lines.append(ln)

    seen = set()
    lines = []
    for ln in raw_lines:
        key = ln.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(ln)

    def is_noise(line_lower: str) -> bool:
        return (
            line_lower.startswith(("pertanyaan", "instruksi", "jawablah", "kamu adalah", "konteks", "riwayat"))
            or "sources:" in line_lower
            or line_lower in ("---",)
            or line_lower.startswith("=== ")
        )

    best, best_score = None, -10**9
    for ln in lines:
        low = ln.lower()
        if is_noise(low):
            continue

        hit = sum(1 for k in kws if k in low)
        length = len(ln)

        bonus = 0
        if 10 <= length <= 60:
            bonus += 8
        if length <= 80:
            bonus += 2
        if ln.endswith("."):
            bonus -= 3
        if re.match(r"^\d+[\.\)]\s", ln):
            bonus -= 2

        penalty = 0
        if length > 80:
            penalty += (length - 80) // 4
        if ln.count(",") >= 3:
            penalty += 3

        score = hit * 10 + bonus - penalty
        if len(kws) >= 2 and hit == 0:
            score -= 6

        if score > best_score:
            best_score = score
            best = ln

    if best:
        best = re.sub(r"\s+", " ", best).strip()
        return (best[:57] + "...") if len(best) > 60 else best

    return (" ".join(kws[:6]))[:60] if kws else "New chat"


def _cache_bust(url: str):
    if not url:
        return ""
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}v=1"


def _clamp_question(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    if len(q) > MAX_QUESTION_CHARS:
        q = q[:MAX_QUESTION_CHARS]
    return q



# App
app = Flask(__name__)

# reverse proxy (cloudflared) – ambil proto & IP dari header
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

app.secret_key = os.environ.get("SECRET_KEY") or "dev-only-change-me"

cookie_secure = (os.environ.get("COOKIE_SECURE", "1") == "1")

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=cookie_secure,
    WTF_CSRF_TIME_LIMIT=None,   # ✅ FIX: None = no expiry
)
app.config["WTF_CSRF_SSL_STRICT"] = False

csrf = CSRFProtect(app)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[RATE_LIMIT_DEFAULT],
)


def _init_state():
    if "active_chat_id" not in session:
        session["active_chat_id"] = None
    if "chats" not in session:
        session["chats"] = {}


def _new_chat(title="New chat"):
    chat_id = str(uuid.uuid4())[:8]
    session["chats"][chat_id] = {
        "title": title,
        "created": datetime.now(timezone.utc).isoformat(),
        "messages": [],
        "state": {"active_person": "", "last_topic": ""},
    }
    session["active_chat_id"] = chat_id
    session.modified = True
    return chat_id


def _get_active_chat():
    _init_state()
    cid = session.get("active_chat_id")
    if cid is None or cid not in session["chats"]:
        _new_chat()
        cid = session["active_chat_id"]

    chat = session["chats"][cid]
    # backward compatibility if older session objects exist
    if "state" not in chat:
        chat["state"] = {"active_person": "", "last_topic": ""}
    if "messages" not in chat:
        chat["messages"] = []

    return cid, chat


def _sorted_chats_items():
    chats_items = list(session["chats"].items())
    chats_items.sort(key=lambda kv: kv[1].get("created", ""), reverse=True)
    return chats_items


@app.after_request
def add_security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' https: data:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; "
        "base-uri 'self'; "
        "frame-ancestors 'none';"
    )
    return resp


HTML = r"""
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{{ app_name }}</title>

  {% if favicon_url %}
    <link rel="icon" href="{{ favicon_url }}" type="image/png">
    <link rel="shortcut icon" href="{{ favicon_url }}" type="image/png">
  {% endif %}

  <style>
    :root{
      --bg: #0b0f14;
      --border: rgba(255,255,255,.08);
      --text: rgba(255,255,255,.88);
      --muted: rgba(255,255,255,.60);
      --muted2: rgba(255,255,255,.42);
      --shadow: 0 18px 60px rgba(0,0,0,.55);
      --r: 14px;

      --sidebar-w: 360px;
      --sidebar-min: 300px;
      --sidebar-max: 520px;
    }
    *{ box-sizing: border-box; }
    html,body{ height:100%; }
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      color: var(--text);
      background: radial-gradient(900px 600px at 20% 10%, rgba(255,255,255,.07), transparent 60%),
                  radial-gradient(700px 500px at 80% 30%, rgba(255,255,255,.05), transparent 60%),
                  var(--bg);
    }

    .app{
      display:grid;
      grid-template-columns: var(--sidebar-w) 6px 1fr;
      height: 100vh;
    }

    .resizer{
      cursor: col-resize;
      background: rgba(255,255,255,.04);
      border-right: 1px solid rgba(255,255,255,.06);
      border-left: 1px solid rgba(255,255,255,.06);
    }
    .resizer:hover{ background: rgba(255,255,255,.07); }

    .sidebar{
      border-right: 1px solid var(--border);
      background: rgba(255,255,255,.02);
      padding: 14px;
      display:flex;
      flex-direction: column;
      gap: 12px;
      min-width: var(--sidebar-min);
    }

    .brand{
      display:flex;
      align-items:center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 10px;
      border: 1px solid var(--border);
      border-radius: var(--r);
      background: rgba(255,255,255,.03);
    }
    .brand-left{
      display:flex;
      align-items:center;
      gap: 10px;
      min-width: 0;
    }

    .brand-logo{
      width: 46px;
      height: 46px;
      border-radius: 0;
      border: none;
      background: transparent;
      overflow: visible;
      display:flex;
      align-items:center;
      justify-content:center;
      flex: 0 0 auto;
    }
    .brand-logo img{
      width: 46px;
      height: 46px;
      object-fit: contain;
      display:block;
      filter: drop-shadow(0 10px 20px rgba(0,0,0,.45));
    }

    .brand-text{ min-width: 0; }
    .brand-name{
      font-weight: 800;
      letter-spacing:.2px;
      line-height: 1.1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 220px;
    }
    .brand-tagline{
      font-size: 12px;
      color: var(--muted2);
      margin-top: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 220px;
    }

    .pill{
      font-size: 12px;
      color: var(--muted);
      border: 1px solid var(--border);
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,.03);
      flex: 0 0 auto;
    }

    .btn{
      width:100%;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(255,255,255,.04);
      color: var(--text);
      cursor:pointer;
      font-weight: 700;
      transition: .15s ease;
    }
    .btn:hover{ background: rgba(255,255,255,.06); transform: translateY(-1px); }

    .list{
      overflow:auto;
      padding-right: 4px;
      display:flex;
      flex-direction: column;
      gap: 8px;
    }

    .chat-row{
      display:flex;
      gap: 8px;
      align-items: stretch;
    }

    .chat-item{
      flex: 1;
      display:flex;
      align-items:center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 10px;
      border: 1px solid transparent;
      border-radius: 12px;
      color: var(--muted);
      background: transparent;
      cursor:pointer;
      transition: .15s ease;
      width: 100%;
      text-align: left;
      min-width: 0;
    }
    .chat-item:hover{
      background: rgba(255,255,255,.04);
      border-color: rgba(255,255,255,.06);
      color: var(--text);
    }
    .chat-item.active{
      background: rgba(255,255,255,.06);
      border-color: rgba(255,255,255,.10);
      color: var(--text);
    }

    .chat-title{
      max-width: none;
      overflow:hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 14px;
      font-weight: 700;
    }
    .chat-meta{
      font-size: 12px;
      color: var(--muted2);
    }

    .chat-actions{
      display:flex;
      align-items:center;
      gap: 6px;
      flex: 0 0 auto;
    }
    .icon-btn{
      border: 1px solid rgba(255,255,255,.10);
      background: rgba(255,255,255,.03);
      color: rgba(255,255,255,.78);
      border-radius: 10px;
      padding: 6px 7px;
      cursor:pointer;
      transition: .15s ease;
      font-weight: 800;
      font-size: 12px;
      white-space: nowrap;
    }
    .icon-btn:hover{
      background: rgba(255,255,255,.06);
      border-color: rgba(255,255,255,.16);
      transform: translateY(-1px);
    }
    .icon-btn.danger{
      color: rgba(255,180,180,.9);
      border-color: rgba(255,120,120,.18);
    }

    .main{
      display:flex;
      flex-direction: column;
      height: 100vh;
    }
    .topbar{
      padding: 14px 18px;
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,.02);
      display:flex;
      align-items:center;
      justify-content: space-between;
      gap: 12px;
    }
    .topbar .title{ font-weight: 800; letter-spacing:.2px; }
    .topbar .sub{
      font-size: 13px;
      color: var(--muted);
      margin-top: 2px;
    }

    .chat{
      flex:1;
      overflow:auto;
      padding: 18px;
    }
    .wrap{
      max-width: 920px;
      margin: 0 auto;
      display:flex;
      flex-direction: column;
      gap: 12px;
    }

    .msg{
      display:flex;
      gap: 10px;
      align-items:flex-start;
    }
    .avatar{
      width: 34px;
      height: 34px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,.04);
      display:flex;
      align-items:center;
      justify-content:center;
      color: rgba(255,255,255,.78);
      font-size: 12px;
      font-weight: 900;
      flex: 0 0 auto;
    }
    .bubble{
      border: 1px solid var(--border);
      background: rgba(255,255,255,.03);
      border-radius: 16px;
      padding: 12px 14px;
      box-shadow: var(--shadow);
      max-width: 100%;
      white-space: pre-wrap;
      line-height: 1.55;
    }
    .msg.user .bubble{ background: rgba(255,255,255,.06); }
    .msg.assistant .bubble{ background: rgba(0,0,0,.18); }

    .composer{
      border-top: 1px solid var(--border);
      background: rgba(255,255,255,.02);
      padding: 12px 18px 16px;
    }
    .composer .wrap{ gap: 10px; }
    .inputrow{
      display:flex;
      gap: 10px;
      align-items:flex-end;
    }
    textarea{
      flex: 1;
      min-height: 52px;
      max-height: 180px;
      resize: vertical;
      padding: 12px 12px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,.25);
      color: var(--text);
      outline: none;
      line-height: 1.5;
      font-size: 14px;
    }
    textarea::placeholder{ color: var(--muted2); }

    .send{
      border-radius: 14px;
      padding: 12px 14px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,.92);
      color: rgba(0,0,0,.85);
      cursor:pointer;
      font-weight: 900;
      transition: .15s ease;
      min-width: 90px;
    }
    .send:hover{ transform: translateY(-1px); background: rgba(255,255,255,.98); }

    .footnote{ color: var(--muted2); font-size: 12px; }

    /* Modal (yang kemarin hilang) */
    .modal-backdrop{
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,.55);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 18px;
      z-index: 50;
    }
    .modal{
      width: min(520px, 100%);
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 16px;
      background: rgba(20,26,34,.96);
      box-shadow: 0 30px 120px rgba(0,0,0,.65);
      padding: 14px;
    }
    .modal h3{ margin: 0 0 8px; font-size: 16px; }
    .modal p{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .modal input{
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,.14);
      background: rgba(0,0,0,.25);
      color: var(--text);
      outline: none;
    }
    .modal-actions{
      display:flex;
      justify-content: flex-end;
      gap: 10px;
      margin-top: 12px;
    }
    .modal-actions .btn{
      width: auto;
      padding: 10px 12px;
    }

    @media (max-width: 900px){
      .app{ grid-template-columns: 1fr; }
      .resizer{ display:none; }
      .sidebar{
        position: sticky;
        top: 0;
        z-index: 5;
        border-right: none;
        border-bottom: 1px solid var(--border);
      }
    }
  </style>
</head>

<body>
<div class="app">

  <aside class="sidebar">
    <div class="brand">
      <div class="brand-left">
        <div class="brand-logo">
          {% if brand_image_url %}
            <img src="{{ brand_image_url }}" alt="Brand"
                 referrerpolicy="no-referrer"
                 onerror="this.style.display='none'; this.parentElement.innerHTML='<span style=\'font-weight:900;\'>AI</span>'; ">
          {% else %}
            <span style="font-weight:900;">AI</span>
          {% endif %}
        </div>

        <div class="brand-text">
          <div class="brand-name">{{ app_name }}</div>
          <div class="brand-tagline">{{ app_tagline }}</div>
        </div>
      </div>
      <div class="pill">Tunnel</div>
    </div>

    <form method="POST" action="/new">
      <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
      <button class="btn" type="submit">+ New chat</button>
    </form>

    <div class="list" aria-label="Chat history">
      {% for cid, c in chats %}
        <div class="chat-row">
          <form method="POST" action="/switch" style="margin:0; flex:1; min-width:0;">
            <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
            <input type="hidden" name="chat_id" value="{{ cid }}">
            <button class="chat-item {% if cid == active_chat_id %}active{% endif %}" type="submit">
              <div style="min-width:0;">
                <div class="chat-title">{{ c.title }}</div>
                <div class="chat-meta">{{ c.created[:10] }}</div>
              </div>
              <div class="chat-meta">›</div>
            </button>
          </form>

          <div class="chat-actions">
            <button class="icon-btn" type="button"
              onclick="openRename('{{ cid }}','{{ (c.title or '')|e }}')">Rename</button>

            <form method="POST" action="/delete" style="margin:0;">
              <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
              <input type="hidden" name="chat_id" value="{{ cid }}">
              <button class="icon-btn danger" type="submit" title="Delete chat">Del</button>
            </form>
          </div>
        </div>
      {% endfor %}
    </div>
  </aside>

  <div class="resizer" id="resizer" title="Tarik untuk memperbesar sidebar"></div>

  <section class="main">
    <div class="topbar">
      <div>
        <div class="title">{{ active_title }}</div>
        <div class="sub">Tanyakan tentang apapun terkait Universitas Indonesia</div>
      </div>
      <form method="POST" action="/clear" style="margin:0;">
        <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
        <button class="btn" type="submit" style="max-width:140px;">Clear chat</button>
      </form>
    </div>

    <div class="chat" id="chat">
      <div class="wrap">
        {% if messages|length == 0 %}
          <div class="msg assistant">
            <div class="avatar">V</div>
            <div class="bubble">
              Halo! Saya Veri, asisten virtual Universitas Indonesia. Ada yang bisa saya bantu?
            </div>
          </div>
        {% endif %}

        {% for m in messages %}
          <div class="msg {{ m.role }}">
            <div class="avatar">{% if m.role == "user" %}YOU{% else %}AI{% endif %}</div>
            <div class="bubble">{{ m.content }}</div>
          </div>
        {% endfor %}
      </div>
    </div>

    <div class="composer">
      <div class="wrap">
        <form method="POST" action="/" class="inputrow">
          <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
          <textarea name="question" id="question" placeholder="Tulis pertanyaan..." autofocus></textarea>
          <button class="send" type="submit">Kirim</button>
        </form>
        <div class="footnote">Jawaban berdasarkan informasi yang tersedia di knowledge base.</div>
      </div>
    </div>
  </section>

</div>

<div class="modal-backdrop" id="renameBackdrop" onclick="closeRename(event)">
  <div class="modal" role="dialog" aria-modal="true" onclick="event.stopPropagation()">
    <h3>Rename chat</h3>
    <p>Ganti nama chat agar mudah dicari.</p>
    <form method="POST" action="/rename" id="renameForm">
      <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
      <input type="hidden" name="chat_id" id="renameChatId">
      <input type="text" name="new_title" id="renameInput" placeholder="Nama chat..." maxlength="60">
      <div class="modal-actions">
        <button class="btn" type="button" onclick="closeRename()">Batal</button>
        <button class="btn" type="submit">Simpan</button>
      </div>
    </form>
  </div>
</div>

<script>
  const chat = document.getElementById('chat');
  chat.scrollTop = chat.scrollHeight;

  const resizer = document.getElementById('resizer');
  let isResizing = false;
  const clamp = (n, min, max) => Math.max(min, Math.min(max, n));

  resizer.addEventListener('mousedown', () => {
    isResizing = true;
    document.body.style.userSelect = 'none';
  });

  window.addEventListener('mousemove', (e) => {
    if (!isResizing) return;
    const min = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--sidebar-min')) || 300;
    const max = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--sidebar-max')) || 520;
    const w = clamp(e.clientX, min, max);
    document.documentElement.style.setProperty('--sidebar-w', w + 'px');
  });

  window.addEventListener('mouseup', () => {
    if (!isResizing) return;
    isResizing = false;
    document.body.style.userSelect = '';
  });

  const backdrop = document.getElementById('renameBackdrop');
  const input = document.getElementById('renameInput');
  const idField = document.getElementById('renameChatId');

  function openRename(chatId, currentTitle){
    idField.value = chatId;
    input.value = currentTitle || "";
    backdrop.style.display = "flex";
    setTimeout(() => input.focus(), 50);
  }
  function closeRename(){
    backdrop.style.display = "none";
  }
</script>
</body>
</html>
"""


@app.post("/new")
@limiter.limit("10/minute")
def new_chat():
    _init_state()
    _new_chat()
    return redirect(url_for("index"))


@app.post("/switch")
@limiter.limit("30/minute")
def switch_chat():
    _init_state()
    chat_id = request.form.get("chat_id")
    if chat_id and chat_id in session["chats"]:
        session["active_chat_id"] = chat_id
        session.modified = True
    return redirect(url_for("index"))


@app.post("/clear")
@limiter.limit("10/minute")
def clear_chat():
    cid, chat = _get_active_chat()
    chat["messages"] = []
    chat["state"] = {"active_person": "", "last_topic": ""}
    session["chats"][cid] = chat
    session.modified = True
    return redirect(url_for("index"))


@app.post("/rename")
@limiter.limit("20/minute")
def rename_chat():
    _init_state()
    chat_id = request.form.get("chat_id")
    new_title = (request.form.get("new_title") or "").strip()
    new_title = re.sub(r"\s+", " ", new_title)[:60]
    if chat_id and chat_id in session["chats"] and new_title:
        session["chats"][chat_id]["title"] = new_title
        session.modified = True
    return redirect(url_for("index"))


@app.post("/delete")
@limiter.limit("10/minute")
def delete_chat():
    _init_state()
    chat_id = request.form.get("chat_id")
    if chat_id and chat_id in session["chats"]:
        del session["chats"][chat_id]
        if session.get("active_chat_id") == chat_id:
            items = _sorted_chats_items()
            if items:
                session["active_chat_id"] = items[0][0]
            else:
                session["active_chat_id"] = None
                _new_chat()
        session.modified = True
    return redirect(url_for("index"))


@app.route("/", methods=["GET", "POST"])
@csrf.exempt
@limiter.limit(RATE_LIMIT_ASK)
def index():
    _init_state()
    cid, chat = _get_active_chat()

    if request.method == "POST":
        q = _clamp_question(request.form.get("question") or "")
        if not q:
            return redirect(url_for("index"))

        chat["messages"].append({"role": "user", "content": q})

        try:
            ans = query_rag(q, chat.get("messages", []), chat.get("state", {}))
        except Exception:
            ans = "Terjadi error saat memproses. Silakan coba lagi."

        if chat["title"] in ("New chat", "", None):
            chat["title"] = _make_title_from_context_or_keywords(q, _LAST_CONTEXT)

        chat["messages"].append({"role": "assistant", "content": ans})
        session["chats"][cid] = chat
        session.modified = True

        return redirect(url_for("index"))

    return render_template_string(
        HTML,
        app_name=APP_NAME,
        app_tagline=APP_TAGLINE,
        brand_image_url=_cache_bust(BRAND_IMAGE_URL),
        favicon_url=_cache_bust(FAVICON_URL),
        chats=_sorted_chats_items(),
        active_chat_id=session["active_chat_id"],
        active_title=chat.get("title", "New chat"),
        messages=chat.get("messages", []),
        csrf_token=generate_csrf(),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=False)
