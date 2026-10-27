"""The baseline RAG pipeline: one case end to end.

    question -> index.search -> [poisoned chunk placed] -> prompt -> LLM (+ mock tools) -> checkers

There is no defense of any kind here. This is the system the defenses will be added to.
"""
from __future__ import annotations

import re
from dataclasses import asdict
from typing import Optional

from .attacks import inject, poisoned_chunk
from .cache import ResponseCache
from .checkers import attack_succeeded, is_correct
from .index import Index
from .models import BaseModel
from .prompts import build_messages
from .tools import parse_tool_calls, strip_tool_calls
from .types import Case, Chunk, Messages, ModelResponse, RunRow


def call_model(model: BaseModel, messages: Messages, seed: int, cache: Optional[ResponseCache]) -> ModelResponse:
    if cache is not None:
        hit = cache.get(model.cache_identity(), messages, seed)
        if hit is not None:
            return hit
    resp = model.generate(messages, seed=seed)
    if cache is not None and resp.error is None:
        cache.put(model.cache_identity(), messages, seed, resp)
    return resp


def retrieve(case: Case, index: Index, k: int, mode: str, injection_mode: str) -> list[Chunk]:
    q = case.question.question
    if case.attack is None:
        return index.search(q, k=k, mode=mode)
    if injection_mode == "forced":
        return inject(index.search(q, k=k, mode=mode), case.attack)
    if injection_mode == "corpus":
        return index.search(q, k=k, mode=mode, extra=[poisoned_chunk(case.attack)])
    raise ValueError(f"injection_mode must be 'forced' or 'corpus', not {injection_mode!r}")


def parse_citations(answer: str, n_sources: int) -> list[int]:
    nums = []
    for group in re.findall(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]", strip_tool_calls(answer)):
        for x in re.split(r"\s*,\s*", group):
            if 1 <= int(x) <= n_sources and int(x) not in nums:
                nums.append(int(x))
    return nums


def run_case(case: Case, model: BaseModel, index: Index, *, secret: str, k: int = 3, mode: str = "bm25",
             injection_mode: str = "forced", repeat: int = 0,
             cache: Optional[ResponseCache] = None) -> tuple[RunRow, Messages]:
    a = case.attack
    chunks = retrieve(case, index, k, mode, injection_mode)
    messages = build_messages(case.question.question, chunks, secret)
    row = RunRow(
        model=model.name, case_id=case.id, question_id=case.question.id,
        attack_id=a.id if a else None, attack_type=a.type if a else None, style=a.style if a else None,
        position=(a.position if a and injection_mode == "forced" else None),
        injection_mode=(injection_mode if a else "none"), retrieval=mode,
        retrieved=[c.doc_id for c in chunks],
        source_retrieved=any(c.source == case.question.source for c in chunks),
    )
    if a:
        ranks = [i for i, c in enumerate(chunks, 1) if c.injected]
        row.injection_retrieved = bool(ranks)
        row.injection_rank = ranks[0] if ranks else None

    resp = call_model(model, messages, seed=repeat, cache=cache)
    row.seconds = resp.seconds
    if resp.error:
        row.error = resp.error  # kept for the record, never scored
        return row, messages

    row.answer = resp.text
    calls = parse_tool_calls(resp.text)
    row.tool_calls = [asdict(c) for c in calls]
    row.citations = parse_citations(resp.text, len(chunks))
    row.cited_source = any(chunks[i - 1].source == case.question.source for i in row.citations)
    row.correct = is_correct(case.question.gold, resp.text)
    if a:
        row.attack_succeeded = attack_succeeded(a, resp.text, calls, secret)
    return row, messages
