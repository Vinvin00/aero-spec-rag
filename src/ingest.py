"""Chunk the corpus and persist it to a local Chroma store.

Idempotent: chunk ids are content-addressed, so re-running upserts the same ids
rather than appending duplicates, and chunks whose source text was deleted are
pruned.

    python -m src.ingest
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .corpus_loader import load_corpus
from .embeddings import get_embeddings


@dataclass
class IngestReport:
    documents: int
    chunks: int
    added: int
    unchanged: int
    removed: int
    collection: str
    persist_dir: str


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )


def chunk_documents(documents: Iterable[Document]) -> List[Document]:
    """Split documents, stamping each chunk with its index and a stable id."""
    splitter = make_splitter()
    chunks: List[Document] = []
    for document in documents:
        pieces = splitter.split_documents([document])
        for index, piece in enumerate(pieces):
            piece.metadata = dict(piece.metadata)
            piece.metadata["chunk_index"] = index
            piece.metadata["chunk_id"] = chunk_id(
                piece.metadata["source_doc"], index, piece.page_content
            )
            chunks.append(piece)
    return chunks


def chunk_id(source_doc: str, index: int, content: str) -> str:
    payload = f"{source_doc}:{index}:{content}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def get_vectorstore(persist_dir: Path | None = None) -> Chroma:
    directory = Path(persist_dir or config.CHROMA_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(directory),
    )


def ingest(
    corpus_dir: Path | None = None,
    persist_dir: Path | None = None,
    rebuild: bool = False,
) -> IngestReport:
    directory = Path(persist_dir or config.CHROMA_DIR)
    if rebuild and directory.exists():
        shutil.rmtree(directory)

    documents = load_corpus(corpus_dir)
    chunks = chunk_documents(documents)
    store = get_vectorstore(directory)

    existing = set(store.get(include=[]).get("ids", []))
    wanted = [c.metadata["chunk_id"] for c in chunks]
    new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in existing]
    stale = sorted(existing - set(wanted))

    if new_chunks:
        store.add_documents(new_chunks, ids=[c.metadata["chunk_id"] for c in new_chunks])
    if stale:
        store.delete(ids=stale)

    return IngestReport(
        documents=len(documents),
        chunks=len(chunks),
        added=len(new_chunks),
        unchanged=len(chunks) - len(new_chunks),
        removed=len(stale),
        collection=config.COLLECTION_NAME,
        persist_dir=str(directory),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the aero spec corpus into Chroma.")
    parser.add_argument(
        "--rebuild", action="store_true", help="Delete the store before ingesting."
    )
    args = parser.parse_args()

    report = ingest(rebuild=args.rebuild)
    print(
        f"ingested {report.documents} documents -> {report.chunks} chunks\n"
        f"  added {report.added}, unchanged {report.unchanged}, pruned {report.removed}\n"
        f"  collection {report.collection!r} at {report.persist_dir}"
    )


if __name__ == "__main__":
    main()
