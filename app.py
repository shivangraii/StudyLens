"""StudyLens - Gradio UI. Run: python app.py"""
from __future__ import annotations

import html

import gradio as gr

from studylens.config import Config
from studylens.engine import StudyLens
from studylens.schemas import Chunk, youtube_id

# --- Hugging Face ZeroGPU compatibility -------------------------------------------------
# ZeroGPU refuses to start a Space that has no @spaces.GPU function. This app does all its work
# on CPU, so we register a do-nothing function just to satisfy that check. Outside ZeroGPU
# (your laptop, CPU Spaces) the `spaces` package is absent and this is a plain function.
try:
    import spaces

    _gpu = spaces.GPU
except ImportError:  # pragma: no cover
    def _gpu(fn):
        return fn


@_gpu
def _zerogpu_registration() -> bool:
    return True


engine: StudyLens | None = None


def get_engine() -> StudyLens:
    global engine
    if engine is None:
        engine = StudyLens(Config())
    return engine


# ------------------------------------------------------------------ helpers
def nb_choices():
    return [(m["name"], nid) for nid, m in get_engine().list_notebooks().items()]


def sources_md(nb_id):
    if not nb_id:
        return "_Create or select a notebook first._"
    srcs = get_engine().list_sources(nb_id)
    if not srcs:
        return "_No sources yet._"
    rows = "\n".join(f"| {s['name']} | {s['type']} | {s['chunks']} |" for s in srcs)
    return f"| Source | Type | Chunks |\n|---|---|---|\n{rows}"


def chat_md(history):
    if not history:
        return "_Ask a question about your sources._"
    return "\n\n---\n\n".join(f"**You:** {h['q']}\n\n**StudyLens:** {h['a']}" for h in history)


def evidence_md(chunk: Chunk) -> str:
    loc = chunk.location_label()
    snippet = chunk.text.replace("\n", " ")
    link = chunk.youtube_link() if chunk.source_type == "video" else None
    head = f"**{chunk.source_name}**" + (f" - {loc}" if loc else "")
    if link:
        head += f"  ([open on YouTube]({link}))"
    return f"{head}\n\n> {snippet[:900]}{'...' if len(snippet) > 900 else ''}"


def player_html(chunk: Chunk) -> str:
    vid = youtube_id(chunk.url)
    if chunk.source_type == "video" and vid and chunk.start_ts is not None:
        src = f"https://www.youtube.com/embed/{vid}?start={int(chunk.start_ts)}&autoplay=1"
        return (
            f'<iframe width="100%" height="260" src="{html.escape(src)}" frameborder="0" '
            'allow="autoplay; encrypted-media" allowfullscreen></iframe>'
        )
    return ""


# ------------------------------------------------------------------ handlers
def refresh_nbs():
    ch = nb_choices()
    return gr.update(choices=ch, value=ch[0][1] if ch else None)


def create_nb(name):
    nid = get_engine().create_notebook(name or "Untitled notebook")
    return gr.update(choices=nb_choices(), value=nid), ""


def on_nb_change(nb_id):
    return sources_md(nb_id), "_Ask a question about your sources._", [], gr.update(choices=[], value=None), "", ""


def _guard(nb_id):
    if not nb_id:
        raise gr.Error("Create or select a notebook first.")


def add_file(nb_id, file):
    _guard(nb_id)
    if file is None:
        raise gr.Error("Choose a file first.")
    path = file if isinstance(file, str) else getattr(file, "name", str(file))
    try:
        name, n = get_engine().add_file(nb_id, path)
    except Exception as e:  # show a readable error instead of a stack trace
        raise gr.Error(str(e))
    return f"Added **{name}** ({n} chunks).", sources_md(nb_id)


def add_youtube(nb_id, url):
    _guard(nb_id)
    if not url.strip():
        raise gr.Error("Paste a YouTube link.")
    try:
        title, n, how = get_engine().add_youtube(nb_id, url.strip())
    except Exception as e:
        raise gr.Error(str(e))
    return f"Added **{title}** ({n} chunks, via {how}).", sources_md(nb_id)


def add_web(nb_id, url):
    _guard(nb_id)
    if not url.strip():
        raise gr.Error("Paste an article link.")
    try:
        title, n = get_engine().add_web(nb_id, url.strip())
    except Exception as e:
        raise gr.Error(str(e))
    return f"Added **{title}** ({n} chunks).", sources_md(nb_id)


def ask(nb_id, question, history, hybrid, rerank, verify):
    _guard(nb_id)
    if not question.strip():
        raise gr.Error("Type a question.")
    try:
        ans = get_engine().ask(nb_id, question.strip(), history, use_bm25=hybrid, use_rerank=rerank, verify=verify)
    except Exception as e:
        raise gr.Error(str(e))

    text = ans.text
    if ans.verification and ans.verification.get("score") is not None:
        v = ans.verification
        flagged = [d["sentence"] for d in v["details"] if not d["supported"]]
        text += f"\n\n_Citation check: {v['score']:.0%} of cited sentences supported._"
        if flagged:
            text += "\n\n_Possibly unsupported:_ " + " | ".join(f"\u26a0\ufe0f {s}" for s in flagged)
    if ans.latency:
        text += f"\n\n<sub>{ans.latency['total']:.1f}s</sub>"

    history = history + [{"q": question.strip(), "a": text}]
    cites = [{"n": c.n, "payload": c.chunk.to_payload()} for c in ans.citations]
    choices = [
        (f"[{c.n}] {c.chunk.source_name} {('· ' + c.chunk.location_label()) if c.chunk.location_label() else ''}", str(c.n))
        for c in ans.citations
    ]
    first = ans.citations[0].chunk if ans.citations else None
    return (
        chat_md(history),
        history,
        "",
        cites,
        gr.update(choices=choices, value=choices[0][1] if choices else None),
        evidence_md(first) if first else "",
        player_html(first) if first else "",
    )


def on_cite_select(n, cites):
    for c in cites or []:
        if str(c["n"]) == str(n):
            chunk = Chunk.from_payload(c["payload"])
            return evidence_md(chunk), player_html(chunk)
    return "", ""


def studio_call(fn_name):
    def _run(nb_id):
        _guard(nb_id)
        try:
            return getattr(get_engine(), fn_name)(nb_id)
        except Exception as e:
            raise gr.Error(str(e))

    return _run


# ------------------------------------------------------------------ UI
def build_app() -> gr.Blocks:
    with gr.Blocks(title="StudyLens") as demo:
        gr.Markdown(
            "# \U0001F4DA StudyLens\nA multimodal study notebook - ask questions across videos, PDFs, slides and web pages "
            "and get **cited** answers (page numbers + video timestamps)."
        )
        history = gr.State([])
        cites = gr.State([])

        with gr.Row():
            # ---------------- left: notebook + sources
            with gr.Column(scale=1):
                nb = gr.Dropdown(label="Notebook", choices=[], interactive=True)
                with gr.Row():
                    new_name = gr.Textbox(label="New notebook", placeholder="e.g. Machine Learning", scale=3)
                    create_btn = gr.Button("Create", scale=1)
                with gr.Tabs():
                    with gr.Tab("File"):
                        file_in = gr.File(label="PDF / PPTX / audio / video / txt", type="filepath")
                        file_btn = gr.Button("Add file")
                    with gr.Tab("YouTube"):
                        yt_in = gr.Textbox(label="YouTube URL")
                        yt_btn = gr.Button("Add video")
                    with gr.Tab("Web"):
                        web_in = gr.Textbox(label="Article URL")
                        web_btn = gr.Button("Add page")
                status = gr.Markdown()
                sources = gr.Markdown("_Create or select a notebook first._")

            # ---------------- middle: chat
            with gr.Column(scale=2):
                chat = gr.Markdown("_Ask a question about your sources._")
                q = gr.Textbox(label="Your question", placeholder="Where does the lecturer explain gradient descent?", lines=2)
                with gr.Row():
                    ask_btn = gr.Button("Ask", variant="primary")
                    clear_btn = gr.Button("Clear chat")
                with gr.Accordion("Retrieval options", open=False):
                    hybrid = gr.Checkbox(value=True, label="Hybrid search (vector + BM25)")
                    rerank = gr.Checkbox(value=True, label="Cross-encoder reranker")
                    verify = gr.Checkbox(value=False, label="Verify citations (extra LLM call)")

            # ---------------- right: evidence + studio
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Evidence"):
                        cite_dd = gr.Dropdown(label="Cited source", choices=[], interactive=True)
                        evidence = gr.Markdown()
                        player = gr.HTML()
                    with gr.Tab("Studio"):
                        sum_btn = gr.Button("Summary")
                        quiz_btn = gr.Button("Quiz (5 MCQs)")
                        card_btn = gr.Button("Flashcards")
                        studio_out = gr.Markdown()

        demo.load(refresh_nbs, None, nb)
        create_btn.click(create_nb, new_name, [nb, new_name])
        nb.change(on_nb_change, nb, [sources, chat, history, cite_dd, evidence, player])
        file_btn.click(add_file, [nb, file_in], [status, sources])
        yt_btn.click(add_youtube, [nb, yt_in], [status, sources])
        web_btn.click(add_web, [nb, web_in], [status, sources])

        ask_in = [nb, q, history, hybrid, rerank, verify]
        ask_out = [chat, history, q, cites, cite_dd, evidence, player]
        ask_btn.click(ask, ask_in, ask_out)
        q.submit(ask, ask_in, ask_out)
        clear_btn.click(lambda: ("_Ask a question about your sources._", []), None, [chat, history])
        cite_dd.change(on_cite_select, [cite_dd, cites], [evidence, player])

        sum_btn.click(studio_call("summarize"), nb, studio_out)
        quiz_btn.click(studio_call("quiz"), nb, studio_out)
        card_btn.click(studio_call("flashcards"), nb, studio_out)
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch()
