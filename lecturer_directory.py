"""Small, audited directory for list-style lecturer questions.

This is not a full faculty/staff directory. We use an explicit, finite list
because semantic nearest-neighbour search cannot reliably enumerate all
members of a university or department.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIRECTORY_JSON = ROOT / "data" / "dosen" / "lecturer_directory_verified.json"


def lecturer_directory_answer(directory_file: Path = DIRECTORY_JSON) -> tuple[str, list[dict]]:
    """Return a disclosed *partial* list plus file+page provenance."""
    if not directory_file.is_file():
        return (
            "Dataset ini belum memiliki daftar dosen terverifikasi yang dapat ditampilkan. "
            "Silakan cek direktori resmi di https://math.ui.ac.id/.",
            [],
        )
    data = json.loads(directory_file.read_text(encoding="utf-8"))
    people = data.get("people", [])
    if not people:
        return "Belum ada profil dosen terverifikasi dalam daftar ini.", []

    lines = ["Berikut beberapa dosen Departemen Matematika FMIPA UI yang tercantum dalam sumber resmi yang sudah kami kumpulkan:"]
    for i, person in enumerate(people, 1):
        lines.append(f"{i}. {person['name']} — {person['field']}")
    lines.append("Ini hanya contoh nama yang terdokumentasi dalam knowledge base, BUKAN daftar lengkap dosen UI atau Departemen Matematika.")
    directory_url = data.get("directory_url")
    if directory_url:
        lines.append(f"Untuk informasi staf terbaru, cek website resmi departemen: {directory_url}")

    pages = sorted({p["page"] for p in people if isinstance(p.get("page"), int) and p["page"] > 0})
    sources = [
        {"source": data["source_pdf"], "page": page, "document_number": i}
        for i, page in enumerate(pages, 1)
    ]
    return "\n".join(lines), sources


def known_lecturer_in_text(text: str, directory_file: Path = DIRECTORY_JSON) -> str:
    """Detect an exact full lecturer name for domain routing (no fuzzy guesses)."""
    if not text or not directory_file.is_file():
        return ""
    import re
    q = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    data = json.loads(directory_file.read_text(encoding="utf-8"))
    for person in data.get("people", []):
        name = person.get("name", "")
        canonical = " ".join(re.findall(r"[a-z0-9]+", name.casefold()))
        if canonical and re.search(r"(?:^|\s)" + re.escape(canonical) + r"(?:$|\s)", q):
            return name
    return ""
