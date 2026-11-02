"""Testable, dependency-light retrieval and grounding utilities for Veritas.

Chroma's similarity_search_with_score returns *distance* (lower is better),
not similarity. The project uses unit-normalized MiniLM embeddings; its legacy
Chroma index uses L2 distance, so tune MAX_DISTANCE on a labelled evaluation set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
import re
from typing import Callable, Iterable, Mapping, Any

log = logging.getLogger(__name__)
NO_EVIDENCE = "Saya tidak menemukan informasi itu di dataset yang tersedia."
DOMAINS = ("umum", "event", "fasilitas", "dosen")


@dataclass(frozen=True)
class Evidence:
    content: str
    domain: str
    distance: float
    source: str = ""
    page: int | None = None  # 1-indexed, user-facing page
    chunk_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.source}, hlm. {self.page}" if self.source and self.page is not None else "Sumber tidak tersedia"


def is_list_fakultas_program_sekolah(question: str) -> bool:
    """Only handle *explicit list* requests covering faculties AND programs/schools.

    'Program studi FMIPA' should be retrieved normally, not intercepted.
    """
    text = re.sub(r"\s+", " ", (question or "").lower()).strip()
    list_terms = ("daftar", "list", "sebutkan", "sebutin", "tuliskan", "apa saja", "apa aja", "semua")
    combined_terms = (
        "fakultas program sekolah", "fakultas, program", "fakultas dan program",
        "fakultas/program", "program dan sekolah", "program/sekolah",
    )
    return any(t in text for t in list_terms) and any(t in text for t in combined_terms)



def lecturer_list_scope(question: str) -> str | None:
    """Route broad lecturer-directory questions before vector search.

    Returns 'ui' for university-wide questions, 'matematika' for the
    specifically supported department, 'other' for other scoped lists,
    or None for an ordinary factual/person/research question.
    """
    q = re.sub(r"\s+", " ", (question or "").lower()).strip()
    if not re.search(r"\b(dosen|pengajar|lecturer)\b", q):
        return None
    # A request for a lecturer by research area is a legitimate RAG query,
    # not a request to enumerate an entire department.
    if re.search(r"\b(yang|meneliti|riset|ahli|fokus|bidang|mengajar|pengampu)\b", q):
        return None
    if not re.search(r"\b(siapa|daftar|list|sebutkan|sebutin|sebut|semua|nama|apa saja|apa aja)\b", q):
        return None
    if re.search(r"\b(matematika|mathematics|math)\b", q):
        return "matematika"
    if re.search(r"\b(fmipa|teknik|fisip|feb|fkg|fasilkom|kedokteran|psikologi|farmasi|hukum|vokasi|fkm|fia|fib|departemen|prodi)\b", q):
        return "other"
    # A named-person question is not a request for every lecturer at UI.
    extra = set(re.findall(r"[a-z]+", q)) - {
        "siapa", "saja", "aja", "dosen", "pengajar", "lecturer", "ui",
        "universitas", "indonesia", "di", "dari", "itu", "ada", "para",
        "daftar", "list", "sebutkan", "sebutin", "sebut", "semua",
        "nama", "apa", "yang", "sih", "dong", "tolong", "terdapat",
    }
    if extra:
        return None
    return "ui"


def is_explicit_followup(question: str) -> bool:
    """Keep self-contained short questions independent of previous turns."""
    q = re.sub(r"\s+", " ", (question or "").lower()).strip()
    if not q:
        return False
    if q in {"lanjut", "lanjutkan", "jelaskan lagi", "lebih detail", "detailnya", "sebutkan lagi", "contohnya", "yang lain", "coba jelaskan"}:
        return True
    if q.startswith(("bagaimana dengan", "kalau yang", "terus yang", "lalu bagaimana")):
        return True
    if re.search(r"\b(beliau|dia|mereka|tersebut|tadi|itu)\b", q):
        # 'Apa itu UI?' is a definition, not a reference to a previous topic.
        return not bool(re.match(r"^(apa itu|siapa itu)\b", q))
    return False

def detect_domain(question: str) -> str:
    q = (question or "").lower()
    if re.search(r"\b(dosen|profesor|pengajar|lecturer)\b", q):
        return "dosen"
    if re.search(r"\b(wisuda|pkkmb|omb|seminar|career fair|dies natalis|lomba|event)\b", q):
        return "event"
    if re.search(r"\b(perpustakaan|rsui|balairung|sport center|bis kuning|asrama|student center|danau|masjid|fasilitas)\b", q):
        return "fasilitas"
    return "umum"


def contextualize_query(question: str, last_topic: str = "", active_person: str = "") -> str:
    """Simple explainable follow-up expansion (no second paid model call)."""
    q = (question or "").strip()
    if not q:
        return ""
    low = q.lower()
    references = re.search(r"\b(itu|tersebut|tsb|dia|beliau|mereka|tadi|nya)\b", low)
    short_followup = is_explicit_followup(q)
    topic = active_person if references and active_person else last_topic
    # A self-contained definition question ("Apa itu UI?") starts a new topic.
    self_contained = bool(re.match(r"^(apa itu|siapa itu|apa yang dimaksud)\b", low))
    if topic and not self_contained and (references or short_followup) and topic.lower() not in low:
        return f"{q} (konteks pembicaraan: {topic})"
    return q


def _score_search(db: Any, question: str, domain: str, k: int, max_distance: float) -> list[Evidence]:
    """Fail closed on missing scores. Never make up perfect (zero) distances."""
    try:
        results = db.similarity_search_with_score(question, k=k)
    except Exception as exc:
        log.warning("Scored retrieval failed in %s: %s", domain, type(exc).__name__)
        return []
    out = []
    for doc, score in results:
        if not isinstance(score, (int, float)) or not (0 <= score <= max_distance):
            continue
        content = (getattr(doc, "page_content", "") or "").strip()
        if not content:
            continue
        meta = getattr(doc, "metadata", None) or {}
        raw_page = meta.get("page")
        try:
            page = int(raw_page) if raw_page is not None else None
            if page is not None and page < 1:
                page = None
        except (TypeError, ValueError):
            page = None
        out.append(Evidence(
            content=content, domain=domain, distance=float(score),
            source=str(meta.get("source") or ""), page=page,
            chunk_id=str(meta.get("chunk_id") or ""), metadata=meta,
        ))
    return out


def retrieve(
    question: str,
    primary_domain: str,
    get_db: Callable[[str], Any],
    available_domains: Iterable[str] = DOMAINS,
    *,
    k: int = 5,
    max_distance: float = 1.2,
    max_chars: int = 11000,
) -> list[Evidence]:
    """Prefer primary domain, use `umum` as fallback; expand to all only if empty."""
    available = set(available_domains)
    primary = primary_domain if primary_domain in available else "umum"
    order = [primary]
    if primary != "umum" and "umum" in available:
        order.append("umum")
    collected: list[Evidence] = []
    for domain in order:
        try:
            collected.extend(_score_search(get_db(domain), question, domain, k, max_distance))
        except Exception as exc:
            log.warning("Unable to query domain %s (%s)", domain, type(exc).__name__)

    # Cross-domain search only when the prioritized domains found nothing.
    if not collected:
        for domain in DOMAINS:
            if domain in available and domain not in order:
                try:
                    collected.extend(_score_search(get_db(domain), question, domain, k, max_distance))
                except Exception as exc:
                    log.warning("Unable to query fallback %s (%s)", domain, type(exc).__name__)

    collected.sort(key=lambda e: (e.distance, e.domain != primary))
    dedup: set[str] = set()
    selected: list[Evidence] = []
    length = 0
    for item in collected:
        fingerprint = hashlib.sha256(re.sub(r"\s+", " ", item.content).strip().encode()).hexdigest()
        if fingerprint in dedup:
            continue
        if length + len(item.content) > max_chars:
            # Skip this chunk: shorter ones might still fit.
            continue
        selected.append(item)
        dedup.add(fingerprint)
        length += len(item.content)
        if len(selected) >= k:
            break
    return selected


def build_context(evidence: list[Evidence]) -> str:
    return "\n\n".join(
        f"[Dokumen {i}] DOMAIN: {e.domain} | SUMBER: {e.label}\n{e.content}"
        for i, e in enumerate(evidence, 1)
    )


def public_sources(evidence: list[Evidence]) -> list[dict]:
    """Show real PDF/page references only, deduplicated and ordered."""
    sources, seen = [], set()
    for i, e in enumerate(evidence, 1):
        if not e.source or e.page is None:
            continue
        key = (e.source, e.page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": e.source, "page": e.page, "document_number": i})
    return sources


def grounded_answer(answer: str, evidence: list[Evidence]) -> str:
    if not evidence:
        return NO_EVIDENCE
    return answer.strip() if answer and answer.strip() else NO_EVIDENCE
