from studylens.generation.answerer import cited_numbers, parse_citations, split_sentences
from studylens.generation.utils import extract_json
from studylens.schemas import Chunk, Retrieved


def _r(i):
    return Retrieved(Chunk(chunk_id=f"s:{i}", notebook_id="n", source_id="s", source_name="x", source_type="pdf", text="t", index=i, page=i), 1.0)


def test_cited_numbers_handles_groups_and_out_of_range():
    assert cited_numbers("A [1]. B [2, 3]. C [9]. D [1][2].", max_n=3) == [1, 2, 3]


def test_parse_citations_maps_to_chunks():
    cites = parse_citations("Fact [2].", [_r(1), _r(2)])
    assert [c.n for c in cites] == [2] and cites[0].chunk.page == 2


def test_split_sentences_and_json_extract():
    assert len(split_sentences("One [1]. Two [2]! Three?")) == 3
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
