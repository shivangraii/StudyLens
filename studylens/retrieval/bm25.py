from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Plus

from ..schemas import Chunk

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Keyword index for one notebook (rebuilt when sources change).

    BM25Plus is used instead of BM25Okapi because Okapi gives zero IDF on tiny corpora
    (a term in half the chunks scores 0), which breaks small notebooks.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Plus([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]
