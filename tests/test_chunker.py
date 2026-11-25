"""Unit tests for the document chunker in app.rag."""
from app.rag import chunk_text


def test_chunk_text_empty_returns_empty():
    assert chunk_text("", "empty.md") == []
    assert chunk_text("   \n\t  ", "blank.md") == []


def test_chunk_text_single_short_doc():
    text = "Northstar Shop offers a 30-day return window."
    chunks = chunk_text(text, "returns_policy.md", chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].doc_name == "returns_policy.md"
    assert chunks[0].chunk_id == 0
    assert "30-day return window" in chunks[0].text


def test_chunk_text_splits_with_overlap():
    words = [f"word{i}" for i in range(120)]
    text = " ".join(words)
    chunks = chunk_text(text, "long_doc.md", chunk_size=50, overlap=10)
    assert len(chunks) == 3
    # Chunk 0: 0..49, Chunk 1: 40..89, Chunk 2: 80..119
    assert chunks[0].chunk_id == 0
    assert chunks[1].chunk_id == 1
    assert chunks[2].chunk_id == 2
    assert "word40" in chunks[0].text
    assert "word40" in chunks[1].text
    assert "word80" in chunks[1].text
    assert "word80" in chunks[2].text


def test_chunk_text_rejects_invalid_overlap():
    try:
        chunk_text("hello world", "bad.md", chunk_size=50, overlap=50)
        assert False, "Expected ValueError when overlap >= chunk_size"
    except ValueError:
        pass
