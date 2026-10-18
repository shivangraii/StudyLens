"""End-to-end smoke test with no model downloads: hashing embedder + fake LLM + a generated PDF."""
import pymupdf
import pytest

from studylens.config import Config
from studylens.engine import StudyLens
from studylens.generation.prompts import ANSWER_SYSTEM, CONDENSE_SYSTEM, VERIFY_SYSTEM


class FakeLLM:
    def chat(self, messages, **kw):
        sys = messages[0]["content"]
        if sys == ANSWER_SYSTEM:
            user = messages[1]["content"]
            if "gradient" in user.split("Question:")[1].lower():
                return "Gradient descent moves parameters against the gradient [1]."
            return "NOT_IN_SOURCES"
        if sys == VERIFY_SYSTEM:
            return '{"results": [{"id": 0, "supported": true}]}'
        if sys == CONDENSE_SYSTEM:
            return "what is gradient descent"
        return "{}"


def make_pdf(path):
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_textbox(pymupdf.Rect(50, 50, 550, 700), "Gradient descent is an optimisation algorithm that updates parameters in the direction opposite to the gradient of the loss. " * 3)
    p2 = doc.new_page()
    p2.insert_textbox(pymupdf.Rect(50, 50, 550, 700), "Photosynthesis converts sunlight water and carbon dioxide into glucose and oxygen inside chloroplasts. " * 3)
    doc.save(path)
    doc.close()


@pytest.fixture()
def engine(tmp_path):
    cfg = Config(data_dir=tmp_path / "data", embed_model="hashing", enable_reranker=False)
    return StudyLens(cfg, llm=FakeLLM())


def test_full_flow(engine, tmp_path):
    pdf = tmp_path / "notes.pdf"
    make_pdf(str(pdf))
    nb = engine.create_notebook("test")
    name, n = engine.add_file(nb, str(pdf))
    assert name == "notes.pdf" and n >= 2
    assert engine.list_sources(nb)[0]["chunks"] == n

    ans = engine.ask(nb, "What is gradient descent?", verify=True)
    assert ans.found and ans.citations
    assert ans.citations[0].chunk.page == 1  # right page cited
    assert ans.verification["score"] == 1.0

    miss = engine.ask(nb, "Who won the 1998 world cup?")
    assert not miss.found

    # re-adding the same file replaces it instead of duplicating
    engine.add_file(nb, str(pdf))
    assert engine.list_sources(nb)[0]["chunks"] == n


def test_retrieval_modes(engine):
    nb = engine.create_notebook("modes")
    engine.add_text(nb, "bio", "Photosynthesis happens in chloroplasts. " * 20)
    engine.add_text(nb, "ml", "Backpropagation computes gradients for neural networks. " * 20)
    for flags in ({"use_bm25": False}, {"use_vector": False}, {}):
        top = engine.retrieve(nb, "chloroplasts photosynthesis", use_rerank=False, top_k=1, **flags)
        assert top[0].chunk.source_name == "bio"
