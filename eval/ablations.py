"""Run every retrieval config and save a comparison table -> eval/results/ablation.csv

    python -m eval.ablations --notebook <id>

Chunk-size ablation: set CHUNK_WORDS=200 / 350 / 600, re-ingest into a fresh notebook each time,
and run this script again (the notebook id differs per chunk size).
"""
from __future__ import annotations

import argparse
import csv

from studylens.config import Config
from studylens.engine import StudyLens

from .common import CONFIGS, load_queries
from .run_retrieval_eval import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", required=True)
    ap.add_argument("--queries", default="eval/datasets/queries.jsonl")
    ap.add_argument("--out", default="eval/results/ablation.csv")
    a = ap.parse_args()

    engine = StudyLens(Config())
    queries = load_queries(a.queries)
    rows = []
    for name, flags in CONFIGS.items():
        r = evaluate(engine, a.notebook, queries, flags)
        rows.append({"config": name, **{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}})
        print(rows[-1])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = list(rows[0])
    print("\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
