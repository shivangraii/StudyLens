"""Plain data classes shared across the whole project."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional


def fmt_ts(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


_YT_RE = re.compile(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})")


def youtube_id(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = _YT_RE.search(url)
    return m.group(1) if m else None


@dataclass
class Chunk:
    chunk_id: str
    notebook_id: str
    source_id: str
    source_name: str
    source_type: str  # pdf | pptx | video | audio | web | text
    text: str
    index: int = 0
    page: Optional[int] = None
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    url: Optional[str] = None

    def to_payload(self) -> dict:
        return asdict(self)

    @classmethod
    def from_payload(cls, p: dict) -> "Chunk":
        return cls(**{k: p.get(k) for k in cls.__dataclass_fields__ if k in p})

    def location_label(self) -> str:
        if self.page is not None:
            return f"p.{self.page}" if self.source_type == "pdf" else f"slide {self.page}"
        if self.start_ts is not None:
            return f"{fmt_ts(self.start_ts)}–{fmt_ts(self.end_ts)}"
        return ""

    def youtube_link(self) -> Optional[str]:
        vid = youtube_id(self.url)
        if vid and self.start_ts is not None:
            return f"https://youtu.be/{vid}?t={int(self.start_ts)}"
        return self.url


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


@dataclass
class Citation:
    n: int
    chunk: Chunk


@dataclass
class Answer:
    text: str
    found: bool = True
    citations: list = field(default_factory=list)  # list[Citation]
    retrieved: list = field(default_factory=list)  # list[Retrieved]
    verification: Optional[dict] = None
    latency: dict = field(default_factory=dict)
