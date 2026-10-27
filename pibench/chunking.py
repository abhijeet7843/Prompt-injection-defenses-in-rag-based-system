"""Reading the corpus and splitting it into chunks.

Each document is a Markdown file whose first line is "# Title". Chunks are built
from whole sentences, packed up to `max_words`, never crossing a document boundary.
Neighbouring chunks share `overlap` sentences, so a fact that sits on a boundary is
still retrievable with its context.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .types import Chunk

# Words after which a full stop does not end a sentence.
_ABBREV = {"dr", "prof", "mr", "mrs", "ms", "no", "st", "vs", "etc", "e.g", "i.e", "b.tech", "m.tech",
           "rs", "approx", "dept", "sr", "jr"}
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Document:
    id: str      # file stem, e.g. "fees"
    title: str
    text: str    # body without the title line


def load_documents(folder: str | Path) -> list[Document]:
    docs = []
    for path in sorted(Path(folder).glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        lines = raw.splitlines()
        if lines and lines[0].startswith("#"):
            title, body = lines[0].lstrip("#").strip(), "\n".join(lines[1:]).strip()
        else:
            title, body = path.stem.replace("_", " ").title(), raw
        docs.append(Document(id=path.stem, title=title, text=body))
    if not docs:
        raise FileNotFoundError(f"No .md documents found in {folder}")
    return docs


def corpus_fingerprint(folder: str | Path) -> str:
    """Changes whenever any document is added, removed or edited."""
    h = hashlib.sha256()
    for path in sorted(Path(folder).glob("*.md")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        pieces = _BOUNDARY.split(para)
        merged: list[str] = []
        for piece in pieces:
            if merged:
                last_word = merged[-1].rsplit(" ", 1)[-1].rstrip(".").lower()
                if last_word in _ABBREV:
                    merged[-1] = f"{merged[-1]} {piece}"
                    continue
            merged.append(piece)
        out.extend(merged)
        out.append("\n")  # paragraph marker, used to prefer paragraph boundaries
    while out and out[-1] == "\n":
        out.pop()
    return out


def chunk_document(doc: Document, max_words: int = 80, overlap: int = 1) -> list[Chunk]:
    units = split_sentences(doc.text)
    sentences = [u for u in units if u != "\n"]
    para_end = set()  # indices of sentences that end a paragraph
    i = -1
    for u in units:
        if u == "\n":
            para_end.add(i)
        else:
            i += 1
    para_end.add(len(sentences) - 1)

    chunks, start = [], 0
    while start < len(sentences):
        end, words = start, 0
        best_break = None
        while end < len(sentences):
            w = len(sentences[end].split())
            if words and words + w > max_words:
                break
            words += w
            if end in para_end:
                best_break = end
            end += 1
        last = end - 1
        # Prefer to stop at a paragraph end if one exists in the second half of the chunk.
        if end < len(sentences) and best_break is not None and best_break >= start + (last - start) // 2:
            last = best_break
        chunks.append(" ".join(sentences[start:last + 1]))
        if last + 1 >= len(sentences):
            break
        start = max(last + 1 - overlap, start + 1)
    return [Chunk(doc_id=f"{doc.id}#{n}", text=t, title=doc.title) for n, t in enumerate(chunks)]


def chunk_corpus(folder: str | Path, max_words: int = 80, overlap: int = 1) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in load_documents(folder):
        chunks.extend(chunk_document(doc, max_words, overlap))
    return chunks
