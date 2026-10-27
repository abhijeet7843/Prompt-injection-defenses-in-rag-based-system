"""The search index over the corpus: BM25 (keywords), dense (embeddings) and hybrid.

  python -m pibench index        build it (or rebuild it if the corpus or settings changed)

Saved under index/ as chunks.jsonl + embeddings.npy + manifest.json. The manifest records
the corpus fingerprint, chunking settings and embedder, so a stale index is never used
silently.

Hybrid search fuses the BM25 and dense rankings with Reciprocal Rank Fusion:
    score(chunk) = sum over rankings of 1 / (rrf_k + rank)
which needs no score calibration between the two methods.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .chunking import chunk_corpus, corpus_fingerprint
from .embeddings import Embedder, build_embedder
from .types import Chunk

MODES = ("bm25", "dense", "hybrid")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the of to in on for and or is are was were be by with at as it its this that these those what when "
    "where which who whom how do does did i my me we our you your can could will would should from into than "
    "then there their they them he she his her any all".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase words without stop words, with a light plural/suffix strip ('books' -> 'book')."""
    out = []
    for w in _WORD.findall(text.lower()):
        if w in _STOP:
            continue
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]  # 'books' -> 'book', but 'campus', 'previous', 'basis' stay whole
        out.append(w)
    return out


def index_text(c: Chunk) -> str:
    """What gets indexed: the title gives every chunk its document's topic."""
    return f"{c.title}. {c.text}" if c.title else c.text


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float64)
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in query:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out


def _ranking(scores: np.ndarray, n: int) -> list[int]:
    """Indices of the top n scores, best first; ties broken by position so results are stable."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return [i for i in order[:n]]


class Index:
    def __init__(self, chunks: list[Chunk], embedder: Optional[Embedder] = None,
                 vectors: Optional[np.ndarray] = None, manifest: Optional[dict] = None,
                 rrf_k: int = 60, candidates: int = 20, cache=None):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = vectors
        self.manifest = manifest or {}
        self.rrf_k, self.candidates = rrf_k, candidates
        self.cache = cache
        self.tokens = [tokenize(index_text(c)) for c in chunks]
        self.bm25 = BM25(self.tokens)

    # ---- building and loading ----
    @classmethod
    def build(cls, cfg: dict[str, Any], cache=None, verbose: bool = True) -> "Index":
        t0 = time.time()
        chunks = chunk_corpus(cfg["corpus_dir"], cfg["chunk_words"], cfg["chunk_overlap"])
        embedder = build_embedder(cfg.get("embedder"))
        vectors = None
        if embedder is not None:
            if verbose:
                print(f"Embedding {len(chunks)} chunks with {embedder.name} ...", flush=True)
            vectors = embedder.embed([index_text(c) for c in chunks], "document", cache)
        manifest = {
            "corpus_fingerprint": corpus_fingerprint(cfg["corpus_dir"]),
            "chunk_words": cfg["chunk_words"], "chunk_overlap": cfg["chunk_overlap"],
            "embedder": embedder.name if embedder else None,
            "n_documents": len({c.source for c in chunks}), "n_chunks": len(chunks),
            "dim": int(vectors.shape[1]) if vectors is not None else None,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "build_seconds": round(time.time() - t0, 1),
        }
        r = cfg.get("retrieval", {})
        return cls(chunks, embedder, vectors, manifest, r.get("rrf_k", 60), r.get("candidates", 20), cache)

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps({"doc_id": c.doc_id, "title": c.title, "text": c.text}, ensure_ascii=False) + "\n")
        if self.vectors is not None:
            np.save(folder / "embeddings.npy", self.vectors)
        elif (folder / "embeddings.npy").exists():
            (folder / "embeddings.npy").unlink()
        (folder / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, cfg: dict[str, Any], cache=None) -> "Index":
        folder = Path(cfg["index_dir"])
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**json.loads(line)) for line in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        vec_path = folder / "embeddings.npy"
        vectors = np.load(vec_path) if vec_path.exists() else None
        r = cfg.get("retrieval", {})
        return cls(chunks, build_embedder(cfg.get("embedder")), vectors, manifest,
                   r.get("rrf_k", 60), r.get("candidates", 20), cache)

    @staticmethod
    def stale_reason(cfg: dict[str, Any]) -> Optional[str]:
        """Why the saved index cannot be used with this config, or None if it is current."""
        path = Path(cfg["index_dir"]) / "manifest.json"
        if not path.exists():
            return "no index has been built yet"
        m = json.loads(path.read_text(encoding="utf-8"))
        emb = build_embedder(cfg.get("embedder"))
        checks = {
            "corpus_fingerprint": corpus_fingerprint(cfg["corpus_dir"]),
            "chunk_words": cfg["chunk_words"], "chunk_overlap": cfg["chunk_overlap"],
            "embedder": emb.name if emb else None,
        }
        for k, v in checks.items():
            if m.get(k) != v:
                return f"{k} changed ({m.get(k)!r} -> {v!r})"
        return None

    @classmethod
    def load_or_build(cls, cfg: dict[str, Any], cache=None, verbose: bool = True) -> "Index":
        reason = cls.stale_reason(cfg)
        if reason is None:
            return cls.load(cfg, cache)
        if verbose:
            print(f"Building the index: {reason}.", flush=True)
        idx = cls.build(cfg, cache, verbose)
        idx.save(cfg["index_dir"])
        return idx

    # ---- searching ----
    def search(self, query: str, k: int = 3, mode: str = "hybrid", extra: Optional[list[Chunk]] = None) -> list[Chunk]:
        """Top k chunks for the query. `extra` chunks (e.g. a poisoned document) are searched as if
        they had been added to the corpus, without changing the saved index."""
        if mode not in MODES:
            raise ValueError(f"retrieval mode must be one of {MODES}")
        if mode in ("dense", "hybrid") and (self.embedder is None or self.vectors is None):
            raise ValueError(f"'{mode}' retrieval needs an embedder in the config and a built index")
        extra = extra or []
        pool = self.chunks + extra
        n = min(len(pool), max(self.candidates, k))

        bm25_rank: list[int] = []
        if mode in ("bm25", "hybrid"):
            bm = BM25(self.tokens + [tokenize(index_text(c)) for c in extra]) if extra else self.bm25
            s = bm.scores(tokenize(query))
            bm25_rank = [i for i in _ranking(s, n) if s[i] > 0]
        dense_rank: list[int] = []
        dense_scores = None
        if mode in ("dense", "hybrid"):
            q = self.embedder.embed([query], "query", self.cache)[0]
            vecs = self.vectors
            if extra:
                vecs = np.vstack([vecs, self.embedder.embed([index_text(c) for c in extra], "document", self.cache)])
            dense_scores = vecs @ q
            dense_rank = _ranking(dense_scores, n)

        if mode == "bm25":
            ranked = [(i, float(s[i])) for i in bm25_rank]
        elif mode == "dense":
            ranked = [(i, float(dense_scores[i])) for i in dense_rank]
        else:
            fused: dict[int, float] = {}
            for ranking in (bm25_rank, dense_rank):
                for r, i in enumerate(ranking, 1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + r)
            ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
        out = []
        for i, score in ranked[:k]:
            c = pool[i]
            out.append(Chunk(doc_id=c.doc_id, text=c.text, title=c.title, score=round(score, 5), injected=c.injected))
        return out

    def info(self) -> dict[str, Any]:
        return dict(self.manifest)


def chunk_to_dict(c: Chunk) -> dict:
    return asdict(c)
