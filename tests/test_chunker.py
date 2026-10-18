from studylens.ingestion.chunker import chunk_pages, chunk_segments, split_words


def test_split_words_overlap():
    text = " ".join(f"w{i}" for i in range(100))
    parts = split_words(text, size=40, overlap=10)
    assert parts[0].split()[0] == "w0"
    assert parts[1].split()[0] == "w30"  # step = 30
    assert parts[-1].split()[-1] == "w99"


def test_split_words_short_and_empty():
    assert split_words("") == []
    assert split_words("a b c", size=10) == ["a b c"]


def test_pages_never_cross_boundary():
    pages = [(1, "alpha " * 100), (2, "beta " * 100)]
    out = chunk_pages(pages, size=60, overlap=10)
    assert {c["page"] for c in out} == {1, 2}
    assert all(("alpha" in c["text"]) != ("beta" in c["text"]) for c in out)


def test_segments_keep_timestamps():
    segs = [{"text": f"s{i}", "start": i * 10.0, "end": i * 10.0 + 10.0} for i in range(20)]
    out = chunk_segments(segs, window=45, overlap=10)
    assert out[0]["start"] == 0.0 and out[0]["end"] >= 45
    assert out[-1]["end"] == 200.0
    # every chunk starts after the previous one started (always advances)
    starts = [c["start"] for c in out]
    assert starts == sorted(set(starts))
