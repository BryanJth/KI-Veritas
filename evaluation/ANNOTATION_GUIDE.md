# How to create valid Veritas evaluation labels

1. Run `python data.py` to index PDFs with source/page metadata.
2. Sample 50–100 real user questions, including answerable, ambiguous, conversational, and intentionally unanswerable questions. The bundled six questions are **unverified examples only**.
3. Inspect relevant PDF pages *independently of the retriever* and manually record the expected answer. Mark unanswerable questions as such and do not invent a supporting chunk.
4. Inspect indexed chunk IDs and record **all** relevant chunk IDs in `relevant_chunk_ids`. These must match the `chunk_id` in stored Chroma metadata. Set `label_status` to `human_verified` only after review.
5. For unanswerable questions, score refusal correctness in a separate human annotation sheet; retrieval metrics requiring nonempty relevant IDs are not defined for them.
6. Run `python eval_rag.py --dataset evaluation/your_verified_questions.jsonl --run-retrieval`. The report includes Recall@K, Precision@K, MRR, and per-question latency for verified queries with ground truth. Unverified questions do **not** count toward headline metrics.
7. For answer correctness/groundedness: run the chatbot on the same questions, manually label each model response (correct/incorrect; supported/unsupported; appropriate refusal yes/no), and report sample sizes and criteria. No answer-quality scores are generated automatically in this version.

The provided template is not a validated benchmark. Avoid publishing scores from it as measured improvements.
