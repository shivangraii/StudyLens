"""Cited answer generation + citation parsing."""
from __future__ import annotations

import re

from ..schemas import Citation, Retrieved
from .llm import LLM
from .prompts import ANSWER_SYSTEM, CONDENSE_SYSTEM, NOT_FOUND

CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def build_context(retrieved: list[Retrieved]) -> str:
    blocks = []
    for i, r in enumerate(retrieved, start=1):
        c = r.chunk
        loc = c.location_label()
        header = f"[{i}] ({c.source_name}{', ' + loc if loc else ''})"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


def cited_numbers(text: str, max_n: int) -> list[int]:
    seen, out = set(), []
    for m in CITE_RE.finditer(text):
        for tok in re.split(r"[,;]\s*", m.group(1)):
            n = int(tok)
            if 1 <= n <= max_n and n not in seen:
                seen.add(n)
                out.append(n)
    return out


def parse_citations(text: str, retrieved: list[Retrieved]) -> list[Citation]:
    return [Citation(n, retrieved[n - 1].chunk) for n in cited_numbers(text, len(retrieved))]


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s]


def condense_question(llm: LLM, question: str, history: list[dict]) -> str:
    if not history:
        return question
    convo = "\n".join(f"User: {h['q']}\nAssistant: {h['a']}" for h in history[-3:])
    out = llm.chat(
        [
            {"role": "system", "content": CONDENSE_SYSTEM},
            {"role": "user", "content": f"History:\n{convo}\n\nLast question: {question}"},
        ],
        temperature=0,
        max_tokens=150,
    )
    return out.strip().strip('"') or question


def generate_answer(llm: LLM, question: str, retrieved: list[Retrieved]) -> tuple[str, bool]:
    """Return (answer_text, found)."""
    user = f"Context passages:\n\n{build_context(retrieved)}\n\nQuestion: {question}"
    out = llm.chat(
        [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=800,
    ).strip()
    if NOT_FOUND in out:
        return "I couldn't find this in your sources.", False
    return out, True
