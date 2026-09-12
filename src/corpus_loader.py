"""Load the markdown corpus into LangChain Documents."""

from __future__ import annotations

from pathlib import Path
from typing import List

import frontmatter
from langchain_core.documents import Document

from . import config

REQUIRED_FIELDS = ("title", "topic", "source_type")
VALID_SOURCE_TYPES = {"illustrative", "derived"}


class CorpusError(RuntimeError):
    """Raised when a corpus document is malformed."""


def corpus_paths(corpus_dir: Path | None = None) -> List[Path]:
    directory = Path(corpus_dir or config.CORPUS_DIR)
    if not directory.is_dir():
        raise CorpusError(f"Corpus directory not found: {directory}")
    return sorted(directory.glob("*.md"))


def load_corpus(corpus_dir: Path | None = None) -> List[Document]:
    """Read every markdown file, validating its YAML frontmatter."""
    documents: List[Document] = []
    for path in corpus_paths(corpus_dir):
        post = frontmatter.load(path)
        missing = [f for f in REQUIRED_FIELDS if not post.metadata.get(f)]
        if missing:
            raise CorpusError(f"{path.name}: missing frontmatter field(s) {missing}")
        source_type = str(post.metadata["source_type"])
        if source_type not in VALID_SOURCE_TYPES:
            raise CorpusError(
                f"{path.name}: source_type {source_type!r} not in {sorted(VALID_SOURCE_TYPES)}"
            )
        if not post.content.strip():
            raise CorpusError(f"{path.name}: empty document body")
        documents.append(
            Document(
                page_content=post.content.strip(),
                metadata={
                    "source_doc": path.name,
                    "title": str(post.metadata["title"]),
                    "topic": str(post.metadata["topic"]),
                    "source_type": source_type,
                },
            )
        )
    if not documents:
        raise CorpusError(f"No markdown documents found in {config.CORPUS_DIR}")
    return documents
