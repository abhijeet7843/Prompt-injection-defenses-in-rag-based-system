"""One SQLite file (results/cache.sqlite) that remembers every model answer and every embedding.

Answers are stored under a hash of (model settings, exact prompt, seed); embeddings under a
hash of (embedder, kind, text). Re-running an experiment only pays for what it has never
seen, so a crash halfway through an overnight run loses nothing, and fixing a checker
never costs new API calls.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from .types import Messages, ModelResponse


def _h(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, text TEXT, seconds REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")
        self.db.commit()

    # ---- model answers ----
    @staticmethod
    def key(model_id: str, messages: Messages, seed: int) -> str:
        return _h(model_id, messages.system, messages.user, seed)

    def get(self, model_id: str, messages: Messages, seed: int) -> Optional[ModelResponse]:
        row = self.db.execute("SELECT text, seconds FROM responses WHERE key = ?",
                              (self.key(model_id, messages, seed),)).fetchone()
        return None if row is None else ModelResponse(text=row[0], seconds=row[1], cached=True)

    def put(self, model_id: str, messages: Messages, seed: int, resp: ModelResponse) -> None:
        self.db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?)",
                        (self.key(model_id, messages, seed), resp.text, resp.seconds))
        self.db.commit()

    # ---- embeddings ----
    def get_vectors(self, embedder_id: str, kind: str, texts: list[str]) -> list[Optional[np.ndarray]]:
        out = []
        for t in texts:
            row = self.db.execute("SELECT dim, vec FROM embeddings WHERE key = ?",
                                  (_h(embedder_id, kind, t),)).fetchone()
            out.append(None if row is None else np.frombuffer(row[1], dtype=np.float32).copy())
        return out

    def put_vectors(self, embedder_id: str, kind: str, texts: list[str], vecs: np.ndarray) -> None:
        self.db.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                            [(_h(embedder_id, kind, t), int(v.shape[0]), v.astype(np.float32).tobytes())
                             for t, v in zip(texts, vecs)])
        self.db.commit()

    def close(self) -> None:
        self.db.close()
