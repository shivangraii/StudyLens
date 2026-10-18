"""Study aids: summary, quiz, flashcards."""
from __future__ import annotations

from ..schemas import Chunk
from .llm import LLM
from .prompts import FLASHCARD_SYSTEM, QUIZ_SYSTEM, SUMMARY_SYSTEM
from .utils import extract_json


def gather_context(chunks: list[Chunk], max_words: int = 5000) -> str:
    """Join chunks in reading order; if too long, sample evenly so every source is represented."""
    total = sum(len(c.text.split()) for c in chunks)
    if total > max_words and chunks:
        stride = total / max_words
        chosen, acc = [], 0.0
        for c in chunks:
            acc += 1
            if acc >= stride:
                chosen.append(c)
                acc -= stride
        chunks = chosen or chunks[:1]
    return "\n\n".join(f"[{c.source_name}] {c.text}" for c in chunks)


def summarize(llm: LLM, chunks: list[Chunk]) -> str:
    return llm.chat(
        [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": gather_context(chunks)}],
        temperature=0.2,
        max_tokens=1000,
    )


def make_quiz(llm: LLM, chunks: list[Chunk], n: int = 5) -> list[dict]:
    raw = llm.chat(
        [{"role": "system", "content": QUIZ_SYSTEM.format(n=n)}, {"role": "user", "content": gather_context(chunks)}],
        temperature=0.4,
        max_tokens=1800,
        json_mode=True,
    )
    data = extract_json(raw)
    qs = data["questions"] if isinstance(data, dict) else data
    return [q for q in qs if len(q.get("options", [])) >= 2 and 0 <= int(q.get("answer_index", -1)) < len(q["options"])]


def make_flashcards(llm: LLM, chunks: list[Chunk], n: int = 10) -> list[dict]:
    raw = llm.chat(
        [{"role": "system", "content": FLASHCARD_SYSTEM.format(n=n)}, {"role": "user", "content": gather_context(chunks)}],
        temperature=0.3,
        max_tokens=1500,
        json_mode=True,
    )
    data = extract_json(raw)
    cards = data["cards"] if isinstance(data, dict) else data
    return [c for c in cards if c.get("front") and c.get("back")]


def quiz_to_markdown(quiz: list[dict]) -> str:
    out = []
    for i, q in enumerate(quiz, start=1):
        opts = "\n".join(f"- {chr(65 + j)}. {o}" for j, o in enumerate(q["options"]))
        ans = chr(65 + int(q["answer_index"]))
        out.append(
            f"**Q{i}. {q['question']}**\n\n{opts}\n\n"
            f"<details><summary>Show answer</summary>\n\n**{ans}** - {q.get('explanation', '')}\n\n</details>\n"
        )
    return "\n".join(out) or "Could not generate a quiz."


def flashcards_to_markdown(cards: list[dict]) -> str:
    rows = "\n".join(
        f"| {c['front'].replace('|', '/')} | <details><summary>flip</summary>{c['back'].replace('|', '/')}</details> |"
        for c in cards
    )
    return f"| Front | Back |\n|---|---|\n{rows}" if cards else "Could not generate flashcards."
