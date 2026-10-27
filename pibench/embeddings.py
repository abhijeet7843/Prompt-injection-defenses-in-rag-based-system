"""Text embedders for dense retrieval. Every embedder returns L2-normalised float32 vectors.

  ollama                 nomic-embed-text through a local Ollama server (default; no extra packages)
  sentence_transformers  any Hugging Face sentence-embedding model (pip install sentence-transformers)
  hash                   an offline, deterministic bag-of-words hash. FOR TESTS ONLY: it knows no
                         meaning, so never report retrieval numbers made with it.

Documents and queries can get different prefixes; nomic-embed-text is trained to expect
"search_document: " and "search_query: ".
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Optional

import numpy as np
import requests


class EmbeddingError(RuntimeError):
    pass


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


class Embedder:
    name = "base"
    doc_prefix = ""
    query_prefix = ""

    def _embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def embed(self, texts: list[str], kind: str = "document", cache=None) -> np.ndarray:
        """kind is "document" or "query". Uses the cache when one is given."""
        prefix = self.doc_prefix if kind == "document" else self.query_prefix
        texts = [prefix + t for t in texts]
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        got = cache.get_vectors(self.name, kind, texts) if cache is not None else [None] * len(texts)
        missing = [i for i, v in enumerate(got) if v is None]
        if missing:
            fresh = _normalise(self._embed([texts[i] for i in missing]))
            if cache is not None:
                cache.put_vectors(self.name, kind, [texts[i] for i in missing], fresh)
            for i, v in zip(missing, fresh):
                got[i] = v
        return _normalise(np.vstack(got))


class OllamaEmbedder(Embedder):
    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434",
                 doc_prefix: Optional[str] = None, query_prefix: Optional[str] = None, batch_size: int = 32,
                 **_: Any):
        self.model, self.host, self.batch_size = model, host.rstrip("/"), batch_size
        nomic = "nomic" in model
        self.doc_prefix = doc_prefix if doc_prefix is not None else ("search_document: " if nomic else "")
        self.query_prefix = query_prefix if query_prefix is not None else ("search_query: " if nomic else "")
        self.name = f"ollama:{model}"

    def _embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                r = requests.post(f"{self.host}/api/embed", json={"model": self.model, "input": batch}, timeout=300)
            except requests.RequestException as e:
                raise EmbeddingError(f"cannot reach Ollama at {self.host} ({e}). Is `ollama serve` running?") from e
            if r.status_code != 200:
                hint = f" Run: ollama pull {self.model}" if "not found" in r.text.lower() else ""
                raise EmbeddingError(f"Ollama embed failed: HTTP {r.status_code}: {r.text[:200]}.{hint}")
            emb = r.json().get("embeddings")
            if not emb or len(emb) != len(batch):
                raise EmbeddingError("Ollama returned no embeddings; check the model is an embedding model")
            rows.extend(emb)
        return np.asarray(rows, dtype=np.float32)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", doc_prefix: str = "", query_prefix: str = "",
                 **_: Any):
        self.model_name, self.doc_prefix, self.query_prefix = model, doc_prefix, query_prefix
        self.name = f"st:{model}"
        self._model = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise EmbeddingError("pip install sentence-transformers to use this embedder") from e
            self._model = SentenceTransformer(self.model_name)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class HashEmbedder(Embedder):
    """Deterministic and offline. Words and word pairs are hashed into `dim` signed buckets."""

    def __init__(self, dim: int = 512, **_: Any):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, t in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", t.lower())
            for feat in words + [f"{a}_{b}" for a, b in zip(words, words[1:])]:
                h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
                m[row, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return m


def build_embedder(spec: Optional[dict[str, Any]]) -> Optional[Embedder]:
    if not spec:
        return None
    spec = dict(spec)
    kind = spec.pop("provider")
    if kind == "ollama":
        return OllamaEmbedder(**spec)
    if kind == "sentence_transformers":
        return SentenceTransformerEmbedder(**spec)
    if kind == "hash":
        return HashEmbedder(**spec)
    raise ValueError(f"Unknown embedder '{kind}'. Use ollama, sentence_transformers or hash.")
