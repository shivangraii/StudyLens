---
title: StudyLens
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.29.0
python_version: "3.12.12"
app_file: app.py
pinned: false
---

# StudyLens - multimodal study notebook (NotebookLM-style)

Ask questions across **YouTube lectures, PDFs, slides, audio and web pages** and get answers with
**verifiable citations**: page numbers for documents, clickable timestamps for videos.
Stack: Python, Gradio, Qdrant (embedded), sentence-transformers, faster-whisper, BM25, cross-encoder reranker.

## How it works

```
INGEST (once per source)                         QUERY (every question)
 PDF / PPTX / web / YouTube / audio              question (+ chat history)
        |  parsers                                      | condense follow-ups (LLM)
        v                                               v
 text + page / timestamps                        vector search (Qdrant) + BM25
        |  chunker (page-aware / time-window)            | Reciprocal Rank Fusion
        v                                               v
 embeddings (bge) --> Qdrant                     cross-encoder rerank -> top 6
                                                        |
                                                        v
                                                LLM answers ONLY from passages, cites [n]
                                                        |
                                                        v
                                                optional citation verification (LLM judge)
                                                        |
                                                        v
                                                UI: answer + evidence + video at timestamp
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # add LLM_API_KEY (free key from groq.com or Google AI Studio)
python app.py                                           # open http://127.0.0.1:7860
```

No-download smoke test: `EMBED_MODEL=hashing ENABLE_RERANKER=0 python -m pytest -q`

## Project layout

| Path | Role |
|---|---|
| `app.py` | Gradio UI (notebooks, add sources, chat, evidence + video player, quiz/flashcards) |
| `studylens/engine.py` | Orchestrator: `add_*`, `retrieve`, `ask`, `summarize/quiz/flashcards` |
| `studylens/config.py` | All settings via env vars |
| `studylens/ingestion/` | `pdf_parser`, `web_parser`, `video_parser` (captions -> whisper fallback), `chunker` |
| `studylens/retrieval/` | `embedder`, `store` (Qdrant), `bm25`, `hybrid` (RRF), `reranker` |
| `studylens/generation/` | `llm` client, `prompts`, `answerer` (cited answers), `citation_check`, `studio` |
| `eval/` | dataset builder, retrieval metrics, ablations, answer-quality eval |
| `scripts/ingest_samples.py` | pre-index demo notebooks from a manifest |
| `tests/` | chunker, RRF/BM25, citation parsing, end-to-end smoke test |

## Evaluation (fill in your numbers)

See `eval/README.md`. Paste results here:

| Config | Recall@5 | MRR |
|---|---|---|
| vector only | | |
| BM25 only | | |
| hybrid | | |
| hybrid + rerank | | |

| Metric | Value |
|---|---|
| Citation hit rate | |
| Faithfulness (LLM-judge) | |
| Refusal accuracy (unanswerable) | |
| Median latency | |

## Deploy on Hugging Face Spaces (live demo)

Free Spaces have an ephemeral disk, so the vector index lives in **Qdrant Cloud** (free tier), not in the Space.

1. cloud.qdrant.io -> create a free cluster -> copy the cluster URL and an API key.
2. Locally, put `QDRANT_URL` and `QDRANT_API_KEY` in `.env`, then pre-index your demo notebooks straight into the cloud:
   `python -m scripts.ingest_samples samples/manifest.json` (YouTube works best from your own laptop, cloud IPs are often blocked).
3. New Space -> SDK **Gradio**, hardware **ZeroGPU** (the only free option; CPU Basic is paid). The app runs entirely on CPU; `app.py` registers a no-op `@spaces.GPU` function only because ZeroGPU requires one. Add this block at the very top of `README.md` (change `sdk_version` to `pip show gradio`):
   ```yaml
   ---
   title: StudyLens
   sdk: gradio
   sdk_version: 6.29.0
   python_version: "3.12.12"
   app_file: app.py
   ---
   ```
4. Upload `app.py`, `requirements.txt`, `packages.txt`, `README.md`, `studylens/`. **Never upload `.env`, `.venv/`, `data/`.**
5. Space -> Settings -> Secrets: `LLM_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`. Variable: `WHISPER_MODEL=base`.
6. Wait for the build (5-10 min). Open the link before sending your resume: free Spaces and free Qdrant clusters both sleep when idle.

Because the index is in a Qdrant server, the app, scripts and eval can all run at the same time. Embedded local
mode (no `QDRANT_URL`) is for development only and allows one process at a time.

## Known limitations (keep this section - honesty reads well)

- Scanned PDFs need OCR first. Slide *images* are not embedded (text + notes only).
- Whisper on free CPU is slow for long videos; prefer captions or pre-indexing.
- Citation verification is an LLM judge, so it has its own error rate.
- Gradio markdown chat is intentionally simple (no streaming).

## Roadmap ideas

SigLIP embeddings for slide frames, source on/off toggles, streaming answers, Hindi (`BAAI/bge-m3`), audio overview (TTS).
