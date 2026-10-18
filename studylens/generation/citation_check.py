"""LLM-judge check: does the cited passage really support each sentence? (the project's differentiator)"""
from __future__ import annotations

import json

from ..schemas import Retrieved
from .answerer import CITE_RE, cited_numbers, split_sentences
from .llm import LLM
from .prompts import VERIFY_SYSTEM
from .utils import extract_json


def verify_citations(llm: LLM, answer_text: str, retrieved: list[Retrieved]) -> dict:
    items = []
    for s in split_sentences(answer_text):
        nums = cited_numbers(s, len(retrieved))
        if not nums:
            continue
        passages = "\n".join(f"[{n}] {retrieved[n - 1].chunk.text}" for n in nums)
        items.append({"id": len(items), "sentence": s, "claim": CITE_RE.sub("", s).strip(), "passages": passages})
    if not items:
        return {"score": None, "details": []}

    payload = [{"id": it["id"], "claim": it["claim"], "passages": it["passages"]} for it in items]
    raw = llm.chat(
        [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        temperature=0,
        max_tokens=600,
        json_mode=True,
    )
    try:
        data = extract_json(raw)
        results = data["results"] if isinstance(data, dict) else data
        flags = {int(r["id"]): bool(r["supported"]) for r in results}
    except (ValueError, KeyError, TypeError):
        return {"score": None, "details": []}

    details = [{"sentence": it["sentence"], "supported": flags.get(it["id"], False)} for it in items]
    score = sum(d["supported"] for d in details) / len(details)
    return {"score": score, "details": details}
