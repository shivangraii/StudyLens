from studylens.retrieval.bm25 import BM25Index
from studylens.retrieval.hybrid import rrf
from studylens.schemas import Chunk


def test_rrf_prefers_docs_in_both_lists():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == "a" and set(ids) == {"a", "b", "c", "d"}
    assert ids.index("c") < ids.index("d")


def _c(i, text):
    return Chunk(chunk_id=f"s:{i}", notebook_id="n", source_id="s", source_name="x", source_type="text", text=text, index=i)


def test_bm25_finds_keyword():
    idx = BM25Index([_c(0, "backpropagation computes gradients"), _c(1, "cats sleep a lot"), _c(2, "rivers flow to the sea")])
    top = idx.search("gradients backpropagation", 2)
    assert top[0][0].chunk_id == "s:0"
