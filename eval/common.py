from __future__ import annotations

import json
from pathlib import Path


def load_queries(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


CONFIGS = {
    "vector_only": dict(use_vector=True, use_bm25=False, use_rerank=False),
    "bm25_only": dict(use_vector=False, use_bm25=True, use_rerank=False),
    "hybrid": dict(use_vector=True, use_bm25=True, use_rerank=False),
    "hybrid+rerank": dict(use_vector=True, use_bm25=True, use_rerank=True),
}
