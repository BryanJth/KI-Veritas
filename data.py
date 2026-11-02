"""Non-destructive, repeatable PDF ingestion for Veritas V2.

By default writes to chroma_verified_2026/; the shipped chroma/ legacy indexes are
never deleted or modified. Use --rebuild only against the V2 destination.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import re
from typing import Iterable

from veritas_core import DOMAINS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict
    id: str


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", name.lower())).strip()


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """Deterministic paragraph-aware splitting, no model dependency.

    Uses hard-length boundaries when necessary, with overlap for continuity.
    """
    if chunk_size < 100 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size >= 100 and 0 <= chunk_overlap < chunk_size required")
    cleaned = re.sub(r"[ \t]+", " ", (text or "")).strip()
    if not cleaned:
        return []
    result = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            # Prefer paragraph/sentence boundary near the end; don't create tiny chunks.
            tail = cleaned[start + chunk_size // 2:end]
            breaks = [tail.rfind("\n\n"), tail.rfind(". "), tail.rfind("\n")]
            boundary = max(breaks)
            if boundary >= 0:
                end = start + chunk_size // 2 + boundary + 1
        segment = cleaned[start:end].strip()
        if segment:
            result.append(segment)
        if end >= len(cleaned):
            break
        start = max(start + 1, end - chunk_overlap)
    return result


def page_domains(relative_path: Path, page_text: str) -> list[str]:
    """Domain folder takes priority; mixed legacy PDF goes to umum + detected tags."""
    if len(relative_path.parts) > 1 and relative_path.parts[0] in DOMAINS:
        return [relative_path.parts[0]]
    text = page_text.lower()
    categories = ["umum"]
    if re.search(r"\b(dosen departemen|daftar dosen|profil dosen)\b", text):
        categories.append("dosen")
    if re.search(r"\b(event dan kegiatan|wisuda|pkkmb|seminar|career fair)\b", text):
        categories.append("event")
    if re.search(r"\b(fasilitas|perpustakaan|balairung|asrama|rsui)\b", text):
        categories.append("fasilitas")
    return categories


def iter_pdf_chunks(pdf_path: Path, data_root: Path, chunk_size: int = 1000, chunk_overlap: int = 150) -> Iterable[Chunk]:
    from pypdf import PdfReader

    relative = pdf_path.relative_to(data_root)
    source_path = relative.as_posix()
    reader = PdfReader(str(pdf_path))
    for page_num, page in enumerate(reader.pages, 1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:
            log.warning("Could not extract %s page %d: %s", source_path, page_num, type(exc).__name__)
            continue
        if len(text) < 20:
            log.info("Skipping empty/image-only PDF page: %s:%d", source_path, page_num)
            continue
        for domain in page_domains(relative, text):
            for i, content in enumerate(split_text(text, chunk_size, chunk_overlap)):
                digest = hashlib.sha256(f"{source_path}|{page_num}|{domain}|{i}|{content}".encode("utf-8")).hexdigest()
                yield Chunk(
                    text=content,
                    metadata={
                        "source": pdf_path.name, "source_path": source_path,
                        "page": page_num, "domain": domain, "chunk_id": digest,
                        "type": "passage",
                    },
                    id=digest,
                )


def discover_pdfs(data_root: Path, domain: str = "all") -> list[Path]:
    pdfs = sorted(data_root.rglob("*.pdf")) if data_root.exists() else []
    if domain == "all":
        return pdfs
    # Root-level mixed-domain PDFs should be indexed when requesting all.
    return [p for p in pdfs if p.parent == data_root / domain]


def index_pdf(db, chunks: list[Chunk], source_path: str) -> tuple[int, int]:
    """Upsert changed/new records; remove stale chunks only for this PDF.

    Never touches other PDFs, nor any legacy index.
    """
    existing = db.get(where={"source_path": source_path}, include=["metadatas"])
    old_ids = set(existing.get("ids") or [])
    new_ids = {c.id for c in chunks}
    delete_ids = list(old_ids - new_ids)
    if delete_ids:
        db.delete(ids=delete_ids)
    fresh = [c for c in chunks if c.id not in old_ids]
    if fresh:
        from langchain_core.documents import Document
        db.add_documents(
            [Document(page_content=c.text, metadata=c.metadata) for c in fresh],
            ids=[c.id for c in fresh],
        )
    return len(fresh), len(delete_ids)


def run(data_root: Path, target_root: Path, domain: str, chunk_size: int, chunk_overlap: int) -> None:
    from langchain_chroma import Chroma
    from get_embedding_function import get_embedding_function

    pdfs = discover_pdfs(data_root, domain)
    if not pdfs:
        raise SystemExit(f"No PDFs found in {data_root}; add documents first.")
    embeddings = get_embedding_function()
    databases = {}
    totals = {d: [0, 0] for d in DOMAINS}
    for pdf in pdfs:
        log.info("Reading %s", pdf)
        chunks = list(iter_pdf_chunks(pdf, data_root, chunk_size, chunk_overlap))
        rel = pdf.relative_to(data_root).as_posix()
        for d in (DOMAINS if domain == "all" else [domain]):
            selected = [c for c in chunks if c.metadata["domain"] == d]
            # Do not create an empty V2 collection which would unintentionally
            # mask an existing legacy collection during the transition.
            if not selected and not (target_root / d / "chroma.sqlite3").is_file():
                continue
            if d not in databases:
                databases[d] = Chroma(
                    collection_name="veritas_" + d,
                    persist_directory=str(target_root / d),
                    embedding_function=embeddings,
                    collection_metadata={"hnsw:space": "l2"},
                )
            new, removed = index_pdf(databases[d], selected, rel)
            totals[d][0] += new
            totals[d][1] += removed
        log.info("%s: %d chunks across %d page-domain groups", rel, len(chunks), len(set(c.metadata['domain'] for c in chunks)))
    for d, (added, removed) in totals.items():
        log.info("%s: %d new chunks; %d stale chunks removed", d, added, removed)
    log.info("Done. Original chroma/ database remains untouched.")


def main() -> None:
    # Match app.py: read CHROMA_PATH from .env before selecting the index target.
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Incremental, source-aware PDF ingestion for Veritas V2")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path(os.getenv("CHROMA_PATH", "chroma_verified_2026")))
    parser.add_argument("--domain", choices=["all", *DOMAINS], default="all")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    args = parser.parse_args()
    if args.output_dir.resolve() == Path("chroma").resolve():
        parser.error("Refusing to write to legacy chroma/. Use chroma_v2/ or another directory.")
    if args.chunk_size < 100 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("Require chunk-size >= 100 and 0 <= chunk-overlap < chunk-size")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.data_dir, args.output_dir, args.domain, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
