"""Loading experiment configs (configs/*.yaml) with sensible defaults."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "corpus_dir": "data/corpus",
    "questions": "data/questions.jsonl",
    "attacks": None,             # a file or list of files; None = clean questions only
    "include_clean": True,
    "index_dir": "index",
    "chunk_words": 80,
    "chunk_overlap": 1,
    "embedder": None,            # None = BM25 only
    "retrieval": {"mode": "bm25", "top_k": 3, "rrf_k": 60, "candidates": 20},
    "injection_mode": "forced",  # forced | corpus  (only used when there are attacks)
    "secret": "NBX-4471-KESTREL",
    "repeats": 1,
    "cache_path": "results/cache.sqlite",
    "max_consecutive_errors": 15,
}


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    for key in ("models", "results_dir"):
        if key not in raw:
            raise ValueError(f"{path} is missing '{key}'")
    cfg = {**DEFAULTS, **raw}
    cfg["retrieval"] = {**DEFAULTS["retrieval"], **(raw.get("retrieval") or {})}
    if cfg["retrieval"]["mode"] in ("dense", "hybrid") and not cfg["embedder"]:
        raise ValueError(f"{path}: retrieval mode '{cfg['retrieval']['mode']}' needs an `embedder:` section")
    if cfg["injection_mode"] not in ("forced", "corpus"):
        raise ValueError(f"{path}: injection_mode must be 'forced' or 'corpus'")
    cfg["_path"] = str(path)
    return cfg


def read_env_text(p: Path) -> str:
    """Read .env whatever encoding Notepad saved it in (UTF-8, UTF-8 with BOM, or UTF-16)."""
    raw = p.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="replace")


def load_env(path: str | Path = ".env") -> list[str]:
    """Minimal .env reader so nobody has to install python-dotenv. Returns the variable names it found."""
    p = Path(path)
    if not p.exists():
        return []
    names = []
    for line in read_env_text(p).splitlines():
        line = line.strip().lstrip("﻿")
        if line.lower().startswith("export "):
            line = line[7:]
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        names.append(k)
        if v:
            os.environ.setdefault(k, v)
    return names
