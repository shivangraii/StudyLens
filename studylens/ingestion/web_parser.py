from __future__ import annotations


def parse_web(url: str) -> tuple[str, str]:
    """Return (title, clean_text) for an article URL using trafilatura."""
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not download {url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    if not text:
        raise ValueError("No readable article text found at that URL.")
    meta = trafilatura.extract_metadata(downloaded)
    title = meta.title if meta and getattr(meta, "title", None) else url
    return title, text
