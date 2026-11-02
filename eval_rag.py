"""Offline retrieval evaluation for Veritas. No Gemini key required.

Ground-truth relevant chunk IDs must come from human review. Seed examples in
`evaluation/qa_template.jsonl` are intentionally *unverified* and excluded.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import time


def compute_metrics(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> dict:
    """Recall@K, Precision@K and reciprocal rank, with valid relevance labels."""
    if k <= 0:
        raise ValueError("k must be positive")
    ranked = retrieved_ids[:k]
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("At least one relevant chunk ID required for ranking metrics")
    hits = len(set(ranked) & relevant)
    rr = next((1.0 / (i + 1) for i, item in enumerate(ranked) if item in relevant), 0.0)
    return {"recall_at_k": hits / len(relevant), "precision_at_k": hits / k, "reciprocal_rank": rr}


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("question") or not row.get("id"):
            raise ValueError(f"{path}:{i} missing id or question")
        rows.append(row)
    return rows


def aggregate(records: list[dict], *, k: int) -> dict:
    valid = [r for r in records if r.get("verified") and r.get("metrics")]
    report = {"k": k, "total_questions": len(records), "verified_scored_questions": len(valid), "unverified_or_unscored": len(records) - len(valid)}
    if valid:
        for metric in ("recall_at_k", "precision_at_k", "reciprocal_rank"):
            report["mean_" + metric] = round(statistics.mean(r["metrics"][metric] for r in valid), 4)
    else:
        report["message"] = "No human-verified ground-truth chunk labels; metrics are intentionally not reported."
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Veritas retrieval evaluation (human-verified labels only)")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/qa_template.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-distance", type=float, default=1.2)
    parser.add_argument("--run-retrieval", action="store_true", help="Load Chroma and execute embedding queries")
    args = parser.parse_args()
    rows = load_dataset(args.dataset)
    records = []
    retriever = None
    if args.run_retrieval:
        # Lazy import: offline dataset validation does not require full AI stack.
        from app import get_db, _available_domains
        from veritas_core import retrieve, detect_domain
        retriever = (get_db, _available_domains, retrieve, detect_domain)

    for row in rows:
        verified = row.get("label_status") == "human_verified"
        gold = row.get("relevant_chunk_ids") or []
        predicted = row.get("retrieved_chunk_ids") or []
        latency = None
        if retriever:
            get_db, available_domains, retrieve, detect_domain = retriever
            start = time.perf_counter()
            chunks = retrieve(row["question"], row.get("domain") or detect_domain(row["question"]), get_db,
                              available_domains(), k=args.k, max_distance=args.max_distance)
            latency = time.perf_counter() - start
            predicted = [e.chunk_id for e in chunks if e.chunk_id]
        metrics = compute_metrics(predicted, gold, args.k) if verified and gold else None
        records.append({"id": row["id"], "question": row["question"], "verified": verified,
                        "retrieved_chunk_ids": predicted, "latency_seconds": round(latency, 4) if latency is not None else None,
                        "metrics": metrics})

    result = {"summary": aggregate(records, k=args.k), "questions": records,
              "limitations": "Retrieval-only; answer faithfulness/correctness require separate human annotations and Gemini calls."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"Saved evaluation report: {args.output}")


if __name__ == "__main__":
    main()
