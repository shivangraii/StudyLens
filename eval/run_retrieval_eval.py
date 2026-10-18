"""Recall@k and MRR for one retrieval configuration.

    python -m eval.run_retrieval_eval --notebook <id> --queries eval/datasets/queries.jsonl --config hybrid+rerank
"""
from __future__ import annotations

import argparse

from studylens.config import Config
from studylens.engine import StudyLens

from .common import CONFIGS, load_queries

KS = (1, 3, 5, 8)


def evaluate(engine: StudyLens, nb_id: str, queries: list[dict], flags: dict, ks=KS) -> dict:
    qs = [q for q in queries if q.get("answerable", True) and q.get("gold_chunk_ids")]
    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    for q in qs:
        got = engine.retrieve(nb_id, q["question"], top_k=max(ks), **flags)
        ids = [r.chunk.chunk_id for r in got]
        gold = set(q["gold_chunk_ids"])
        rank = next((i for i, cid in enumerate(ids, start=1) if cid in gold), None)
        if rank:
            rr_sum += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits[k] += 1
    n = max(1, len(qs))
    out = {f"recall@{k}": hits[k] / n for k in ks}
    out["mrr"] = rr_sum / n
    out["n_queries"] = len(qs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", required=True)
    ap.add_argument("--queries", default="eval/datasets/queries.jsonl")
    ap.add_argument("--config", default="hybrid+rerank", choices=list(CONFIGS))
    a = ap.parse_args()
    engine = StudyLens(Config())
    res = evaluate(engine, a.notebook, load_queries(a.queries), CONFIGS[a.config])
    print(a.config, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})


if __name__ == "__main__":
    main()
