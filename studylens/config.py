"""Central configuration. Everything is overridable via environment variables / .env"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Config:
    # --- storage ---
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))
    # If QDRANT_URL is set, a real Qdrant server (e.g. Qdrant Cloud free tier) is used: safe for many
    # processes/users and data survives restarts. If empty, embedded local mode (single process only).
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", ""))
    qdrant_api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY", ""))

    # --- LLM (any OpenAI-compatible endpoint) ---
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "llama-3.3-70b-versatile"))

    # --- models ---
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "BAAI/bge-reranker-base"))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "small"))
    device: str = field(default_factory=lambda: _env("DEVICE", "cpu"))  # keep "cpu": also required on ZeroGPU
    whisper_compute: str = field(default_factory=lambda: _env("WHISPER_COMPUTE", "int8"))
    enable_reranker: bool = field(default_factory=lambda: _env("ENABLE_RERANKER", "1") == "1")

    # --- chunking ---
    chunk_words: int = field(default_factory=lambda: int(_env("CHUNK_WORDS", "350")))
    chunk_overlap_words: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP_WORDS", "50")))
    video_window_sec: float = field(default_factory=lambda: float(_env("VIDEO_WINDOW_SEC", "45")))
    video_overlap_sec: float = field(default_factory=lambda: float(_env("VIDEO_OVERLAP_SEC", "10")))

    # --- retrieval ---
    retrieve_k: int = 30  # candidates pulled from each retriever
    final_k: int = 6  # passages given to the LLM
    rrf_k: int = 60  # reciprocal-rank-fusion constant

    @property
    def qdrant_path(self) -> str:
        return str(self.data_dir / "qdrant")

    @property
    def notebooks_file(self) -> Path:
        return self.data_dir / "notebooks.json"

    @property
    def query_prefix(self) -> str:
        # BGE English models are trained with this instruction prefix for queries.
        name = self.embed_model.lower()
        if "bge" in name and "m3" not in name and "en" in name:
            return "Represent this sentence for searching relevant passages: "
        return ""
