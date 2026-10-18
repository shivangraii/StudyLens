# Evaluation

1. Ingest 30-50 sources into one notebook (note the notebook id from the UI / `data/notebooks.json`).
2. `python -m eval.build_dataset --notebook <id> --n 200` -> draft questions.
3. Review `eval/datasets/queries_draft.jsonl` by hand, fix bad questions, add cross-source + unanswerable ones,
   save as `eval/datasets/queries.jsonl`.
4. `python -m eval.ablations --notebook <id>` -> Recall@k / MRR table for vector vs BM25 vs hybrid vs hybrid+rerank.
5. `python -m eval.run_answer_eval --notebook <id>` -> citation hit rate, faithfulness, refusal accuracy, latency.
6. Paste the tables into the main README (these numbers are what go on your resume).

Query schema (one JSON per line):
`{"id": "q001", "question": "...", "gold_chunk_ids": ["<source_id>:<index>"], "answerable": true, "type": "single|cross"}`
