"""Corpus loading, chunking, and ingestion idempotency."""

from pathlib import Path

import pytest

from src import config
from src.corpus_loader import CorpusError, VALID_SOURCE_TYPES, corpus_paths, load_corpus
from src.ingest import chunk_documents, ingest


def test_corpus_has_documents():
    paths = corpus_paths()
    assert len(paths) >= 6, "corpus should hold at least 6 markdown documents"


def test_every_document_has_valid_frontmatter():
    documents = load_corpus()
    for document in documents:
        assert document.metadata["title"]
        assert document.metadata["topic"]
        assert document.metadata["source_type"] in VALID_SOURCE_TYPES
        assert document.page_content.strip()


def test_chunking_respects_configured_size():
    chunks = chunk_documents(load_corpus())
    assert chunks, "chunking produced no output"
    for chunk in chunks:
        assert len(chunk.page_content) <= config.CHUNK_SIZE + config.CHUNK_OVERLAP
        assert chunk.metadata["chunk_id"]
        assert chunk.metadata["chunk_index"] >= 0


def test_chunk_ids_are_unique_and_stable():
    first = [c.metadata["chunk_id"] for c in chunk_documents(load_corpus())]
    second = [c.metadata["chunk_id"] for c in chunk_documents(load_corpus())]
    assert first == second, "chunk ids must be deterministic"
    assert len(set(first)) == len(first), "chunk ids must be unique"


def test_ingest_is_idempotent():
    first = ingest()
    second = ingest()
    assert second.chunks == first.chunks
    assert second.added == 0, "a second ingest must not add duplicate chunks"
    assert second.removed == 0
    assert second.unchanged == second.chunks


def test_missing_corpus_directory_raises(tmp_path: Path):
    with pytest.raises(CorpusError):
        load_corpus(tmp_path / "does-not-exist")


def test_malformed_frontmatter_raises(tmp_path: Path):
    (tmp_path / "bad.md").write_text("---\ntitle: No topic\n---\n\nbody\n")
    with pytest.raises(CorpusError):
        load_corpus(tmp_path)
