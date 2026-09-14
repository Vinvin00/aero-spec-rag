"""Embedding backends, selected by AERO_EMBEDDINGS.

* ``fastembed`` (default): BAAI/bge-small-en-v1.5, a learned 384-dim sentence
  embedder run locally on CPU via ONNX. No API key or server; the model is
  downloaded once (~70 MB) and cached (FASTEMBED_CACHE_PATH).
* ``local``: a deterministic hashed bag-of-n-grams, zero download, for fully
  air-gapped runs. Weak at paraphrase.
* ``ollama``: any embedding model served by a local Ollama.

The store is dimension-specific, so re-ingest with --rebuild after switching.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
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


class FastEmbedEmbeddings(Embeddings):
    """Learned sentence embeddings via fastembed (ONNX, CPU)."""

    def __init__(self, model_name: str = config.FASTEMBED_MODEL) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache(maxsize=None)
def _fastembed(model_name: str) -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name)


def get_embeddings() -> Embeddings:
    """Return the embedding backend named by AERO_EMBEDDINGS."""
    backend = config.EMBEDDING_BACKEND
    if backend == "fastembed":
        return _fastembed(config.FASTEMBED_MODEL)
    if backend == "local":
        return HashingEmbeddings()
    if backend == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=config.OLLAMA_EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
    raise ValueError(f"Unknown AERO_EMBEDDINGS backend: {backend!r}")
