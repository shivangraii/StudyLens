"""Chunking strategies. Pure Python -> easy to unit test and to ablate."""
from __future__ import annotations


def split_words(text: str, size: int = 350, overlap: int = 50) -> list[str]:
    """Sliding word window. Returns [] for empty text."""
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = max(1, size - overlap)
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            break
        i += step
    return out


def chunk_pages(pages: list[tuple[int, str]], size: int = 350, overlap: int = 50) -> list[dict]:
    """Page-aware chunking: a chunk never crosses a page boundary, so page citations stay exact."""
    out = []
    for page_no, text in pages:
        for piece in split_words(text, size, overlap):
            out.append({"text": piece, "page": page_no})
    return out


def chunk_text(text: str, size: int = 350, overlap: int = 50) -> list[dict]:
    return [{"text": t} for t in split_words(text, size, overlap)]


def chunk_segments(segments: list[dict], window: float = 45.0, overlap: float = 10.0) -> list[dict]:
    """Time-window chunking for transcripts.

    segments: [{"text", "start", "end"}] sorted by start.
    Each chunk keeps start/end so a citation can jump to the exact moment.
    """
    segs = [s for s in segments if s["text"].strip()]
    if not segs:
        return []
    out, i, n = [], 0, len(segs)
    while i < n:
        start = segs[i]["start"]
        j, texts, end = i, [], segs[i]["end"]
        while j < n:
            texts.append(segs[j]["text"].strip())
            end = segs[j]["end"]
            j += 1
            if end - start >= window:
                break
        out.append({"text": " ".join(texts), "start": start, "end": end})
        if j >= n:
            break
        k = i + 1
        while k < j and segs[k]["start"] < end - overlap:
            k += 1
        i = max(k, i + 1)
    return out
