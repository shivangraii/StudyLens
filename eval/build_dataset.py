"""Draft an eval set: an LLM writes one question per sampled chunk. YOU then review it by hand.

    python -m eval.build_dataset --notebook <id> --n 200

Output: eval/datasets/queries_draft.jsonl  (copy to queries.jsonl after reviewing/editing)
Row schema:
  {"id": "q001", "question": "...", "gold_chunk_ids": ["abc123:4"], "answerable": true, "type": "single"}
Add by hand: ~30 cross-source questions (type "cross", several gold ids) and ~20 unanswerable ones
(answerable=false, gold_chunk_ids=[]).
"""
from __future__ import annotations

import argparse
import json
import random

from studylens.config import Config
from studylens.engine import StudyLens
from studylens.generation.prompts import QUESTION_GEN_SYSTEM
from studylens.generation.utils import extract_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="eval/datasets/queries_draft.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    engine = StudyLens(Config(), reranker=None)
    pool = [c for c in engine.all_chunks(a.notebook) if len(c.text.split()) >= 60]
    random.Random(a.seed).shuffle(pool)
    rows = []
    for c in pool[: a.n]:
        try:
            raw = engine.llm.chat(
                [{"role": "system", "content": QUESTION_GEN_SYSTEM}, {"role": "user", "content": c.text}],
                temperature=0.7,
                max_tokens=120,
                json_mode=True,
            )
            q = extract_json(raw)["question"]
        except Exception as e:
            print("skip:", e)
            continue
        rows.append({"id": f"q{len(rows) + 1:03d}", "question": q, "gold_chunk_ids": [c.chunk_id], "answerable": True, "type": "single", "source_type": c.source_type})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} draft queries -> {a.out}. Now review them manually!")


if __name__ == "__main__":
    main()
