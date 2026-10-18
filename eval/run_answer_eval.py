"""End-to-end answer quality: citation hit rate, faithfulness (LLM-judge), refusal accuracy, latency.

    python -m eval.run_answer_eval --notebook <id> --queries eval/datasets/queries.jsonl --limit 100
"""
from __future__ import annotations

import argparse
import json
import statistics as st

from studylens.config import Config
from studylens.engine import StudyLens

from .common import CONFIGS, load_queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", required=True)
    ap.add_argument("--queries", default="eval/datasets/queries.jsonl")
    ap.add_argument("--config", default="hybrid+rerank", choices=list(CONFIGS))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="eval/results/answers.jsonl")
    a = ap.parse_args()

    engine = StudyLens(Config())
    queries = load_queries(a.queries)[: a.limit]
    cite_hit, faith, false_refusal, refusal_ok, lat = [], [], [], [], []
    with open(a.out, "w") as f:
        for q in queries:
            ans = engine.ask(a.notebook, q["question"], verify=True, **CONFIGS[a.config])
            lat.append(ans.latency.get("total", 0))
            if q.get("answerable", True):
                gold = set(q.get("gold_chunk_ids", []))
                false_refusal.append(not ans.found)
                if ans.found:
                    cite_hit.append(any(c.chunk.chunk_id in gold for c in ans.citations))
                    if ans.verification and ans.verification["score"] is not None:
                        faith.append(ans.verification["score"])
            else:
                refusal_ok.append(not ans.found)
            f.write(json.dumps({"id": q["id"], "question": q["question"], "answer": ans.text, "found": ans.found,
                                "cited": [c.chunk.chunk_id for c in ans.citations],
                                "verification": ans.verification}, ensure_ascii=False) + "\n")

    def mean(x):
        return round(sum(x) / len(x), 3) if x else None

    print({
        "config": a.config,
        "n": len(queries),
        "citation_hit_rate": mean(cite_hit),
        "faithfulness": mean(faith),
        "false_refusal_rate": mean(false_refusal),
        "refusal_accuracy_on_unanswerable": mean(refusal_ok),
        "latency_p50_s": round(st.median(lat), 2) if lat else None,
    })


if __name__ == "__main__":
    main()
