"""Embedding models. `Embedder` = real model, `HashingEmbedder` = offline dev/test mode."""
from __future__ import annotations

import hashlib
import re

import numpy as np


class Embedder:
    def __init__(self, model_name: str, query_prefix: str = "", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.query_prefix = query_prefix
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._load().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    def embed_query(self, query: str) -> list[float]:
        vec = self._load().encode(self.query_prefix + query, normalize_embeddings=True)
        return vec.tolist()


class HashingEmbedder:
    """Bag-of-words hashing vectors. No downloads; good enough to smoke-test the whole pipeline."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) % 2 == 0 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, query: str) -> list[float]:
        return self._vec(query).tolist()


def make_embedder(cfg):
    if cfg.embed_model.lower() == "hashing":
        return HashingEmbedder()
    return Embedder(cfg.embed_model, cfg.query_prefix, cfg.device)
