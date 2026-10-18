"""StudyLens engine: the single object the UI / eval scripts talk to.

    add_* methods  -> ingestion flow  (parse -> chunk -> embed -> store)
    ask()          -> query flow      (condense -> retrieve -> rerank -> cited answer -> verify)
    summarize/quiz/flashcards -> study aids
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from .config import Config
from .generation import answerer, studio
from .generation.citation_check import verify_citations
from .generation.llm import LLM
from .ingestion import chunker
from .ingestion.pdf_parser import parse_pdf, parse_pptx
from .ingestion.video_parser import parse_youtube, transcribe
from .ingestion.web_parser import parse_web
from .retrieval.bm25 import BM25Index
from .retrieval.embedder import make_embedder
from .retrieval.hybrid import rrf
from .retrieval.reranker import Reranker
from .retrieval.store import VectorStore, open_client
from .schemas import Answer, Chunk, Retrieved

MEDIA_EXT = {".mp3", ".wav", ".m4a", ".mp4", ".mkv", ".webm", ".ogg", ".flac", ".mov"}


def _sid(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha1(data).hexdigest()[:12]


class StudyLens:
    def __init__(self, cfg: Optional[Config] = None, *, embedder=None, reranker=None, llm=None):
        self.cfg = cfg or Config()
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or make_embedder(self.cfg)
        if reranker is not None:
            self.reranker = reranker
        else:
            self.reranker = Reranker(self.cfg.rerank_model, self.cfg.device) if self.cfg.enable_reranker else None
        self.llm = llm or LLM(self.cfg.llm_api_key, self.cfg.llm_base_url, self.cfg.llm_model)
        self.store = VectorStore(open_client(self.cfg), self.embedder.dim)
        self.store.import_legacy_notebooks(self.cfg.notebooks_file)
        self._lock = threading.Lock()
        # notebook id -> (chunk count when built, chunks, BM25 index). The count lets us notice
        # that another process (e.g. an ingest script) changed the notebook and rebuild.
        self._cache: dict[str, tuple[int, list[Chunk], BM25Index]] = {}

    # ------------------------------------------------------------------ notebooks
    def list_notebooks(self) -> dict:
        return self.store.list_notebooks()

    def create_notebook(self, name: str) -> str:
        name = name.strip() or "Untitled notebook"
        nb_id = uuid.uuid4().hex[:8]
        self.store.put_notebook(nb_id, name)
        return nb_id

    def delete_notebook(self, nb_id: str) -> None:
        with self._lock:
            self.store.delete_notebook_chunks(nb_id)
            self.store.delete_notebook_meta(nb_id)
            self._cache.pop(nb_id, None)

    # ------------------------------------------------------------------ ingestion
    def _ingest(self, nb_id: str, source_id: str, name: str, stype: str, pieces: list[dict], url: Optional[str] = None) -> int:
        if not pieces:
            raise ValueError("Nothing to index - the source produced no text.")
        chunks = [
            Chunk(
                chunk_id=f"{source_id}:{i}",
                notebook_id=nb_id,
                source_id=source_id,
                source_name=name,
                source_type=stype,
                text=p["text"],
                index=i,
                page=p.get("page"),
                start_ts=p.get("start"),
                end_ts=p.get("end"),
                url=url,
            )
            for i, p in enumerate(pieces)
        ]
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        with self._lock:
            self.store.delete_source(nb_id, source_id)  # re-adding the same source replaces it
            self.store.upsert(chunks, vectors)
            self._cache.pop(nb_id, None)
        return len(chunks)

    def add_file(self, nb_id: str, path: str) -> tuple[str, int]:
        p = Path(path)
        ext = p.suffix.lower()
        data = p.read_bytes()
        sid = _sid(data)
        cfg = self.cfg
        if ext == ".pdf":
            pieces = chunker.chunk_pages(parse_pdf(str(p)), cfg.chunk_words, cfg.chunk_overlap_words)
            stype = "pdf"
        elif ext == ".pptx":
            pieces = chunker.chunk_pages(parse_pptx(str(p)), cfg.chunk_words, cfg.chunk_overlap_words)
            stype = "pptx"
        elif ext in MEDIA_EXT:
            segs = transcribe(p, cfg)
            pieces = chunker.chunk_segments(segs, cfg.video_window_sec, cfg.video_overlap_sec)
            stype = "audio"
        elif ext in {".txt", ".md"}:
            pieces = chunker.chunk_text(data.decode("utf-8", errors="ignore"), cfg.chunk_words, cfg.chunk_overlap_words)
            stype = "text"
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        n = self._ingest(nb_id, sid, p.name, stype, pieces)
        return p.name, n

    def add_youtube(self, nb_id: str, url: str) -> tuple[str, int, str]:
        title, segs, how = parse_youtube(url, self.cfg)
        pieces = chunker.chunk_segments(segs, self.cfg.video_window_sec, self.cfg.video_overlap_sec)
        n = self._ingest(nb_id, _sid(url), title, "video", pieces, url=url)
        return title, n, how

    def add_web(self, nb_id: str, url: str) -> tuple[str, int]:
        title, text = parse_web(url)
        pieces = chunker.chunk_text(text, self.cfg.chunk_words, self.cfg.chunk_overlap_words)
        n = self._ingest(nb_id, _sid(url), title, "web", pieces, url=url)
        return title, n

    def add_text(self, nb_id: str, name: str, text: str) -> int:
        pieces = chunker.chunk_text(text, self.cfg.chunk_words, self.cfg.chunk_overlap_words)
        return self._ingest(nb_id, _sid(name + text), name, "text", pieces)

    def list_sources(self, nb_id: str) -> list[dict]:
        seen: dict[str, dict] = {}
        for c in self.all_chunks(nb_id):
            s = seen.setdefault(c.source_id, {"source_id": c.source_id, "name": c.source_name, "type": c.source_type, "chunks": 0})
            s["chunks"] += 1
        return list(seen.values())

    def delete_source(self, nb_id: str, source_id: str) -> None:
        with self._lock:
            self.store.delete_source(nb_id, source_id)
            self._cache.pop(nb_id, None)

    # ------------------------------------------------------------------ retrieval
    def _load(self, nb_id: str) -> tuple[list[Chunk], BM25Index]:
        n = self.store.count(nb_id)
        hit = self._cache.get(nb_id)
        if hit is None or hit[0] != n:
            chunks = self.store.all_chunks(nb_id)
            hit = (n, chunks, BM25Index(chunks))
            self._cache[nb_id] = hit
        return hit[1], hit[2]

    def all_chunks(self, nb_id: str) -> list[Chunk]:
        return self._load(nb_id)[0]

    def retrieve(
        self, nb_id: str, query: str, *, use_vector: bool = True, use_bm25: bool = True, use_rerank: bool = True, top_k: Optional[int] = None
    ) -> list[Retrieved]:
        cfg = self.cfg
        top_k = top_k or cfg.final_k
        ranked_lists, by_id = [], {}
        if use_vector:
            hits = self.store.search(self.embedder.embed_query(query), nb_id, cfg.retrieve_k)
            ranked_lists.append([c.chunk_id for c, _ in hits])
            by_id.update({c.chunk_id: c for c, _ in hits})
        if use_bm25:
            hits = self._load(nb_id)[1].search(query, cfg.retrieve_k)
            ranked_lists.append([c.chunk_id for c, _ in hits])
            by_id.update({c.chunk_id: c for c, _ in hits})
        fused = rrf(ranked_lists, cfg.rrf_k)[: cfg.retrieve_k]
        cands = [Retrieved(by_id[cid], score) for cid, score in fused]
        if use_rerank and self.reranker is not None and cands:
            return self.reranker.rerank(query, cands, top_k)
        return cands[:top_k]

    # ------------------------------------------------------------------ question answering
    def ask(
        self,
        nb_id: str,
        question: str,
        history: Optional[list[dict]] = None,
        *,
        use_vector: bool = True,
        use_bm25: bool = True,
        use_rerank: bool = True,
        verify: bool = False,
        top_k: Optional[int] = None,
    ) -> Answer:
        t0 = time.time()
        search_q = answerer.condense_question(self.llm, question, history or [])
        t1 = time.time()
        retrieved = self.retrieve(nb_id, search_q, use_vector=use_vector, use_bm25=use_bm25, use_rerank=use_rerank, top_k=top_k)
        t2 = time.time()
        if not retrieved:
            return Answer("This notebook has no sources yet - add a PDF, video or link first.", found=False)
        text, found = answerer.generate_answer(self.llm, search_q, retrieved)
        t3 = time.time()
        ans = Answer(text=text, found=found, retrieved=retrieved)
        if found:
            ans.citations = answerer.parse_citations(text, retrieved)
            if verify:
                ans.verification = verify_citations(self.llm, text, retrieved)
        ans.latency = {"condense": t1 - t0, "retrieve": t2 - t1, "generate": t3 - t2, "total": time.time() - t0}
        return ans

    # ------------------------------------------------------------------ study aids
    def summarize(self, nb_id: str) -> str:
        return studio.summarize(self.llm, self._need_chunks(nb_id))

    def quiz(self, nb_id: str, n: int = 5) -> str:
        return studio.quiz_to_markdown(studio.make_quiz(self.llm, self._need_chunks(nb_id), n))

    def flashcards(self, nb_id: str, n: int = 10) -> str:
        return studio.flashcards_to_markdown(studio.make_flashcards(self.llm, self._need_chunks(nb_id), n))

    def _need_chunks(self, nb_id: str) -> list[Chunk]:
        chunks = self.all_chunks(nb_id)
        if not chunks:
            raise ValueError("Add at least one source first.")
        return chunks
