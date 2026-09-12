"""Embedding backends.

The default backend is deliberately offline: a deterministic hashed bag-of-
n-grams projected into a fixed-dimension unit vector. It needs no API key and no
model download, so ingestion, the graph, and the whole test suite run anywhere.
It is weaker than a learned embedding at paraphrase matching, which is why the
retriever in this project pairs it with a lexical rescoring pass.

Set AERO_EMBEDDINGS=openai or =voyage (plus the matching API key) to swap in a
learned model; nothing else in the pipeline changes.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

from langchain_core.embeddings import Embeddings

from . import config

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> List[str]:
    """Lowercase word/number tokens, keeping decimal numbers intact."""
    return _TOKEN_RE.findall(text.lower())


def _features(text: str) -> List[str]:
    """Unigrams plus bigrams, so short phrases carry a little word order."""
    tokens = tokenize(text)
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


def _bucket(feature: str, dim: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


class HashingEmbeddings(Embeddings):
    """Deterministic, dependency-free, offline embedding function."""

    def __init__(self, dim: int = config.EMBEDDING_DIM) -> None:
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dim
        for feature in _features(text):
            # Sub-linear term weighting; the sign spreads collisions out.
            index = _bucket(feature, self.dim)
            sign = 1.0 if _bucket("s:" + feature, 2) == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            # Chroma rejects all-zero vectors under cosine distance.
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embeddings() -> Embeddings:
    """Return the embedding backend named by AERO_EMBEDDINGS."""
    backend = config.EMBEDDING_BACKEND
    if backend == "local":
        return HashingEmbeddings()
    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")
    if backend == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(model="voyage-3")
    raise ValueError(f"Unknown AERO_EMBEDDINGS backend: {backend!r}")
