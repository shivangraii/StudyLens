from __future__ import annotations

from ..schemas import Retrieved


class Reranker:
    """Cross-encoder reranker (reads query + passage together -> more precise than embeddings)."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, items: list[Retrieved], top_k: int) -> list[Retrieved]:
        if not items:
            return []
        scores = self._load().predict([(query, r.chunk.text) for r in items])
        ranked = sorted(zip(items, scores), key=lambda x: float(x[1]), reverse=True)
        return [Retrieved(r.chunk, float(s)) for r, s in ranked[:top_k]]
