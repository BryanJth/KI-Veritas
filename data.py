# data.py
import os
import re
import shutil
import unicodedata
from typing import List, Tuple, Optional

# Chroma import
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function

try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    PyPDFLoader = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


PDF_PATH = "data/DATASET KA-I.pdf"

CHROMA_UMUM = "chroma/umum"
CHROMA_DOSEN = "chroma/dosen"
CHROMA_EVENT = "chroma/event"
CHROMA_FASILITAS = "chroma/fasilitas"


EXTRA_TEXT_PATHS = [
    "data/extra_kb.txt",
    "data/extra_fakultas.txt",
]


def reset_db():
    for p in [CHROMA_UMUM, CHROMA_DOSEN, CHROMA_EVENT, CHROMA_FASILITAS]:
        if os.path.exists(p):
            shutil.rmtree(p)



# Normalizers
_DEGREE_TOKENS = {
    # common titles / degrees (after punctuation removal: "Ph.D." -> "ph d", "M.Sc." -> "m sc")
    "phd", "ph.d", "ph", "d",
    "m.sc", "msc", "m", "sc",
    "m.si", "msi", "si",
    "dea", "mt", "s.si", "ssi", "s.sos", "s.t", "st",
    "rer", "nat", "dra", "ir", "apt", "mkom", "m.kom",
    "prof", "dr",
}

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_name(name: str) -> str:
    """Lowercase, remove punctuation & degrees so 'Rahmi Rusin, Ph.D.' -> 'rahmi rusin'."""
    n = normalize_text(name)
    parts = [p for p in n.split() if p not in _DEGREE_TOKENS]
    return " ".join(parts).strip()



# PDF loading
def _load_pdf_text_pypdf(pdf_path: str) -> str:
    if PdfReader is None:
        return ""
    reader = PdfReader(pdf_path)
    out = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        out.append(t)
    return "\n".join(out)

def _load_pdf_text_langchain(pdf_path: str) -> str:
    if PyPDFLoader is None:
        return ""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join(p.page_content for p in pages)

def load_pdf_text() -> str:
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF tidak ditemukan: {PDF_PATH}")

    # Try langchain loader first, fallback to pypdf
    text = _load_pdf_text_langchain(PDF_PATH)
    if text and len(text.strip()) > 50:
        return text

    text2 = _load_pdf_text_pypdf(PDF_PATH)
    return text2 or text


def _read_extra_texts(paths: List[str]) -> List[str]:
    texts = []
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    t = f.read().strip()
                if t:
                    texts.append(t)
                    print(f"✅ Loaded extra KB text: {p} ({len(t)} chars)")
            except Exception as e:
                print(f"⚠️ Failed reading {p}: {e}")
    return texts



# Section bounds helper (case-insensitive)
def _find_section_bounds(text: str, start_markers, end_markers) -> Tuple[int, int]:
    lower = text.lower()

    start_idx = -1
    for s in start_markers:
        i = lower.find(s.lower())
        if i != -1:
            start_idx = i
            break

    if start_idx == -1:
        return -1, -1

    end_idx = -1
    for e in end_markers:
        j = lower.find(e.lower(), start_idx + 1)
        if j != -1:
            end_idx = j
            break

    if end_idx == -1:
        end_idx = len(text)

    return start_idx, end_idx


# DOSEN extraction (bounded section + safe numbering)
def _parse_numbered_profiles(section_text: str):
    """
    Parse ONLY 1-2 digit numbering to avoid years like '2018.' being misread as a new profile.
      1. Name
      <profile...>
      2. Name
      <profile...>

    Returns: list of dict {idx, nama, desc}
    """
    results = []

    # IMPORTANT: \d{1,2} prevents 2018. from being captured as a "new person"
    header_re = re.compile(r"(?m)^\s*(\d{1,2})\.\s+([^\n]{2,120})\s*$")
    matches = list(header_re.finditer(section_text))

    if not matches:
        return []

    for i, m in enumerate(matches):
        idx = int(m.group(1))
        nama = m.group(2).strip().strip(",")
        desc_start = m.end()
        desc_end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
        desc_raw = section_text[desc_start:desc_end].strip()

        desc = re.sub(r"\s+", " ", desc_raw).strip()
        if not desc:
            continue

        results.append({"idx": idx, "nama": nama, "desc": f"{nama}. {desc}"})

    return results


def extract_dosen(text: str):
    """
    Extract lecturer profiles section only.
    Start marker: 'DOSEN DEPARTEMEN MATEMATIKA'
    End marker: first 'FASILITAS' after it (so facilities numbering doesn't leak into dosen list).
    """
    start_markers = [
        "DOSEN DEPARTEMEN MATEMATIKA",
        "DAFTAR DOSEN DEPARTEMEN MATEMATIKA",
        "DAFTAR DOSEN",
    ]
    end_markers = [
        "\nFASILITAS",
        "FASILITAS",
        "\nEVENT",
        "EVENT",
        "\nKEBIJAKAN",
        "KEBIJAKAN",
    ]

    s, e = _find_section_bounds(text, start_markers, end_markers)
    if s == -1:
        return []

    section = text[s:e]
    results = _parse_numbered_profiles(section)

    # de-dup by normalized name, keep smallest idx
    best = {}
    for r in results:
        key = normalize_name(r["nama"])
        if not key:
            continue
        if key not in best or r["idx"] < best[key]["idx"]:
            best[key] = r

    final = list(best.values())
    final.sort(key=lambda x: x["idx"])
    return final


def index_dosen(dosen):
    db = Chroma(
        persist_directory=CHROMA_DOSEN,
        embedding_function=get_embedding_function()
    )

    docs = []
    for d in dosen:
        idx = int(d.get("idx", 0) or 0)
        nama_raw = d.get("nama", "").strip()
        nama_norm = normalize_name(nama_raw)

        content = (
            f"Nama: {nama_raw}\n"
            f"Profil: {d.get('desc','').strip()}"
        )

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "idx": idx,
                    "nama": nama_raw,
                    "nama_norm": nama_norm,
                    "domain": "dosen",
                    "type": "person",
                }
            )
        )

    if docs:
        db.add_documents(docs)
    print(f"✅ Indexed {len(docs)} dosen")



# UMUM / event / fasilitas
def extract_umum(text: str) -> Optional[str]:
    s = text.lower().find("profil universitas indonesia")
    e = text.lower().find("kebijakan kekayaan intelektual")
    if s == -1 or e == -1 or e <= s:
        return None
    return text[s:e]


def extract_fakultas_section(text: str) -> Optional[str]:
    """
    Kalau di PDF kamu ada bagian "DAFTAR SEMUA FAKULTAS/PROGRAM/SEKOLAH UI ...",
    ambil section itu biar bisa di-index dan dipanggil bot.

    Kalau tidak ketemu, return None (biar keliatan bahwa PDF kamu memang tidak punya text-nya,
    atau section-nya berupa gambar/tabel yang tidak kebaca extractor).
    """
    start_markers = [
        "DAFTAR SEMUA FAKULTAS",
        "DAFTAR FAKULTAS/PROGRAM/SEKOLAH",
        "DAFTAR FAKULTAS",
        "FAKULTAS/PROGRAM/SEKOLAH",
    ]
    end_markers = [
        "PROFIL UNIVERSITAS INDONESIA",
        "KODE ETIK",
        "KEBIJAKAN",
        "DOSEN",
        "EVENT",
        "FASILITAS",
    ]

    s, e = _find_section_bounds(text, start_markers, end_markers)
    if s == -1:
        return None
    section = text[s:e].strip()
    if len(section) < 80:
        return None
    return section


def extract_by_keywords(text: str, keywords: List[str]) -> List[str]:
    """
    Case-insensitive slicing by keyword positions.
    """
    lower = text.lower()
    sections = []
    for k in keywords:
        i = lower.find(k.lower())
        if i != -1:
            sections.append((i, k))

    if not sections:
        return []

    sections.sort()
    result = []

    for i, (start, _k) in enumerate(sections):
        end = sections[i + 1][0] if i + 1 < len(sections) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            result.append(chunk)

    return result


def index_domain(texts: List[str], path: str, domain: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=160
    )

    docs = []
    for t in texts:
        if not t or len(t.strip()) < 20:
            continue
        for c in splitter.split_text(t):
            docs.append(
                Document(
                    page_content=c,
                    metadata={"domain": domain}
                )
            )

    db = Chroma(
        persist_directory=path,
        embedding_function=get_embedding_function()
    )
    if docs:
        db.add_documents(docs)
    print(f"✅ Indexed {len(docs)} chunks → {domain}")


def debug_faculty_mentions(text: str):
    """
    Cek cepat: apakah PDF kamu memang mengandung banyak baris 'Fakultas ...' sebagai teks?
    Kalau output-nya cuma sedikit, berarti wajar bot tidak bisa 'list 14' dari KB.
    """
    lines = []
    for ln in (text or "").splitlines():
        if re.search(r"\bfakultas\b", ln, flags=re.I):
            ln2 = re.sub(r"\s+", " ", ln).strip()
            if ln2 and ln2 not in lines:
                lines.append(ln2)

    print(f"🔎 Lines containing 'fakultas' (unique, up to 30): {min(len(lines),30)} / {len(lines)}")
    for ln in lines[:30]:
        print("  -", ln)


# MAIN
if __name__ == "__main__":
    print("Resetting chroma DBs...")
    reset_db()

    print("Loading PDF text...")
    text = load_pdf_text()
    print(f"Loaded text length: {len(text)} chars")

    # Debug dulu: apakah 'fakultas' kebaca sebagai teks?
    debug_faculty_mentions(text)

    print("Extracting dosen profiles...")
    dosen_list = extract_dosen(text)
    print(f"Found {len(dosen_list)} dosen profiles. Sample:")
    for d in dosen_list[:10]:
        print(f"  {d['idx']}. {d['nama']}")

    index_dosen(dosen_list)

    umum_section = extract_umum(text)
    fakultas_section = extract_fakultas_section(text)

    extra_texts = _read_extra_texts(EXTRA_TEXT_PATHS)

    umum_texts = []
    if umum_section:
        umum_texts.append(umum_section)
    else:
        print("⚠️ Warning: 'umum' section not found (marker mismatch)")

    if fakultas_section:
        print("✅ Found fakultas section via markers, adding to UMUM index.")
        umum_texts.append(fakultas_section)
    else:
        print("⚠️ Fakultas section marker not found (atau terlalu pendek / tidak kebaca sebagai teks).")

    if extra_texts:
        print("✅ Adding extra KB texts to UMUM index.")
        umum_texts.extend(extra_texts)

    if umum_texts:
        index_domain(umum_texts, CHROMA_UMUM, "umum")

    index_domain(
        extract_by_keywords(text, [
            "EVENT DAN KEGIATAN",
            "WISUDA",
            "ORIENTASI MAHASISWA",
            "SEMINAR",
            "CAREER FAIR"
        ]),
        CHROMA_EVENT,
        "event"
    )

    index_domain(
        extract_by_keywords(text, [
            "FASILITAS",
            "PERPUSTAKAAN",
            "BALAIRUNG",
            "RUMAH SAKIT",
            "SPORT CENTER"
        ]),
        CHROMA_FASILITAS,
        "fasilitas"
    )

    print("Indexing complete.")
