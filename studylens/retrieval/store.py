"""Qdrant vector store + notebook registry.

Two modes (chosen by config):
  * server mode  - QDRANT_URL set (Qdrant Cloud free tier / Docker). Many processes and users can
                   use it at once, and the data survives restarts (needed for Hugging Face Spaces).
  * local mode   - embedded, data lives in a folder. Fast to start, but Qdrant allows only ONE
                   process to open that folder at a time.
Notebook names live in a small 'notebooks' collection in the same Qdrant, so there is no local
state file to lose on an ephemeral host.
"""
from __future__ import annotations

import json
import time
import warnings
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from ..schemas import Chunk

COLLECTION = "chunks"
NOTEBOOKS = "notebooks"


def _pid(notebook_id: str, chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{notebook_id}/{chunk_id}"))


def _nb_filter(notebook_id: str, source_id: Optional[str] = None) -> qm.Filter:
    must = [qm.FieldCondition(key="notebook_id", match=qm.MatchValue(value=notebook_id))]
    if source_id:
        must.append(qm.FieldCondition(key="source_id", match=qm.MatchValue(value=source_id)))
    return qm.Filter(must=must)


def open_client(cfg) -> QdrantClient:
    if cfg.qdrant_url:
        return QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key or None, timeout=60)
    try:
        return QdrantClient(path=cfg.qdrant_path)
    except RuntimeError as e:
        if "already accessed" in str(e):
            raise RuntimeError(
                "Another process (the running app, an ingest script or an eval script) already has the local "
                f"Qdrant folder '{cfg.qdrant_path}' open. Close it first, or set QDRANT_URL/QDRANT_API_KEY "
                "to use a Qdrant server (Qdrant Cloud free tier) that many processes can share."
            ) from e
        raise


class VectorStore:
    def __init__(self, client: QdrantClient, dim: int):
        self.client = client
        self.dim = dim
        if not client.collection_exists(COLLECTION):
            client.create_collection(COLLECTION, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))
        else:
            existing = client.get_collection(COLLECTION).config.params.vectors.size
            if existing != dim:
                raise RuntimeError(
                    f"Stored vectors have dim={existing} but the embedding model gives dim={dim}. "
                    "You changed EMBED_MODEL - delete the collection/folder and re-ingest."
                )
        for key in ("notebook_id", "source_id"):  # speeds up filters on a server; harmless locally
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # local mode warns that indexes have no effect
                    client.create_payload_index(COLLECTION, field_name=key, field_schema=qm.PayloadSchemaType.KEYWORD)
            except Exception:
                pass
        if not client.collection_exists(NOTEBOOKS):
            client.create_collection(NOTEBOOKS, vectors_config=qm.VectorParams(size=1, distance=qm.Distance.COSINE))

    # ---------------------------------------------------------------- chunks
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        points = [
            qm.PointStruct(id=_pid(c.notebook_id, c.chunk_id), vector=np.asarray(v).tolist(), payload=c.to_payload())
            for c, v in zip(chunks, vectors)
        ]
        for i in range(0, len(points), 128):
            self.client.upsert(COLLECTION, points=points[i : i + 128])

    def search(self, vector: list[float], notebook_id: str, limit: int) -> list[tuple[Chunk, float]]:
        res = self.client.query_points(
            COLLECTION, query=vector, query_filter=_nb_filter(notebook_id), limit=limit, with_payload=True
        )
        return [(Chunk.from_payload(p.payload), float(p.score)) for p in res.points]

    def count(self, notebook_id: str) -> int:
        return int(self.client.count(COLLECTION, count_filter=_nb_filter(notebook_id), exact=True).count)

    def all_chunks(self, notebook_id: str) -> list[Chunk]:
        out, offset = [], None
        flt = _nb_filter(notebook_id)
        while True:
            points, offset = self.client.scroll(
                COLLECTION, scroll_filter=flt, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            out.extend(Chunk.from_payload(p.payload) for p in points)
            if offset is None:
                break
        out.sort(key=lambda c: (c.source_name, c.index))
        return out

    def delete_source(self, notebook_id: str, source_id: str) -> None:
        self.client.delete(COLLECTION, points_selector=qm.FilterSelector(filter=_nb_filter(notebook_id, source_id)))

    def delete_notebook_chunks(self, notebook_id: str) -> None:
        self.client.delete(COLLECTION, points_selector=qm.FilterSelector(filter=_nb_filter(notebook_id)))

    # ---------------------------------------------------------------- notebook registry
    def list_notebooks(self) -> dict:
        out, offset = {}, None
        while True:
            points, offset = self.client.scroll(NOTEBOOKS, limit=256, offset=offset, with_payload=True, with_vectors=False)
            for p in points:
                out[p.payload["id"]] = {"name": p.payload["name"], "created": p.payload.get("created", 0)}
            if offset is None:
                break
        return dict(sorted(out.items(), key=lambda kv: kv[1]["created"]))

    def put_notebook(self, nb_id: str, name: str, created: Optional[float] = None) -> None:
        self.client.upsert(
            NOTEBOOKS,
            points=[
                qm.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"notebook/{nb_id}")),
                    vector=[1.0],
                    payload={"id": nb_id, "name": name, "created": created or time.time()},
                )
            ],
        )

    def delete_notebook_meta(self, nb_id: str) -> None:
        self.client.delete(NOTEBOOKS, points_selector=qm.PointIdsList(points=[str(uuid.uuid5(uuid.NAMESPACE_URL, f"notebook/{nb_id}"))]))

    def import_legacy_notebooks(self, json_file: Path) -> None:
        """One-time migration from the old data/notebooks.json registry."""
        if not json_file.exists() or self.list_notebooks():
            return
        for nb_id, meta in json.loads(json_file.read_text()).items():
            self.put_notebook(nb_id, meta["name"], meta.get("created"))
