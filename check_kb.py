"""Inspect verified knowledge base paths without loading models or exposing keys."""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
root = Path(__file__).resolve().parent
index_path = os.environ.get("CHROMA_PATH", "chroma_verified_2026")
index_root = Path(index_path)
if not index_root.is_absolute():
    index_root = root / index_root
verified_only = os.getenv("VERIFIED_ONLY", "1") == "1"
print("Veritas knowledge base diagnostics")
print("Project directory:", root)
print("CHROMA_PATH:", index_root)
print("VERIFIED_ONLY:", int(verified_only))
print("PDF files:", len(list((root / "data").rglob("*.pdf"))))
missing = []
for domain in ("umum", "event", "fasilitas", "dosen"):
    db = index_root / domain / "chroma.sqlite3"
    available = db.is_file() and db.stat().st_size > 0
    print(f"  {domain}: {'index found' if available else 'NOT INDEXED'}")
    if not available:
        missing.append(domain)
if missing:
    print("Action: Run `python data.py --domain all`, then restart `python app.py`.")
if not verified_only:
    print("Warning: VERIFIED_ONLY=0 permits legacy retrieval. Set VERIFIED_ONLY=1 for verified-only answers.")
