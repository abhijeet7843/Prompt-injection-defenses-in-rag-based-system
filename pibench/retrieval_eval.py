"""How good is retrieval on its own, before any model sees the context?

For each question, a chunk counts as relevant if it comes from the question's source
document AND contains one of the gold answers. Reported per retrieval mode:

  Recall@k  share of questions with a relevant chunk in the top k
  MRR@10    mean of 1/rank of the first relevant chunk (0 if not in the top 10)
"""
from __future__ import annotations

from .checkers import contains_phrase
from .index import MODES, Index
from .types import Question


def first_relevant_rank(index: Index, q: Question, mode: str, depth: int = 10) -> int | None:
    for rank, c in enumerate(index.search(q.question, k=depth, mode=mode), 1):
        if c.source == q.source and any(contains_phrase(c.text, g) for g in q.gold):
            return rank
    return None


def evaluate(index: Index, questions: dict[str, Question], ks=(1, 3, 5)) -> tuple[list[dict], list[dict]]:
    modes = [m for m in MODES if m == "bm25" or index.vectors is not None]
    summary, per_question = [], []
    for mode in modes:
        ranks = {}
        for q in questions.values():
            r = first_relevant_rank(index, q, mode)
            ranks[q.id] = r
            per_question.append({"mode": mode, "question_id": q.id, "rank": r})
        n = len(ranks)
        row = {"mode": mode, "n_questions": n}
        for k in ks:
            row[f"recall@{k}"] = sum(1 for r in ranks.values() if r is not None and r <= k) / n
        row["mrr@10"] = sum(1 / r for r in ranks.values() if r) / n
        summary.append(row)
    return summary, per_question


def format_table(summary: list[dict]) -> str:
    keys = [k for k in summary[0] if k.startswith("recall@") or k == "mrr@10"]
    head = f"{'mode':<8}" + "".join(f"{k:>11}" for k in keys)
    lines = [head, "-" * len(head)]
    for row in summary:
        lines.append(f"{row['mode']:<8}" + "".join(f"{row[k]:>11.3f}" for k in keys))
    return "\n".join(lines)
