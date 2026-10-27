"""Checking the automatic scorers against a human reading of real outputs.

1. python -m pibench sample --n 60      -> results/<exp>/handcheck.csv
2. Open the CSV in Excel or Google Sheets. For each row, fill human_attack_succeeded
   (attacked rows only) and human_correct with y or n, reading only the question,
   the injected text and the answer. Do not look at the auto_ columns while labelling.
3. python -m pibench agreement          -> agreement, Cohen's kappa and every disagreement

Report the kappa values. If a checker is wrong often, fix its rule in checkers.py
and run the same experiment again: every answer comes from the cache, so rescoring
needs no new model calls.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from .types import Attack, Question

YES = {"y", "yes", "1", "true", "t"}
NO = {"n", "no", "0", "false", "f"}


def make_sample(rows: list[dict], questions: dict[str, Question], attacks: dict[str, Attack],
                n: int = 60, seed: int = 7) -> list[dict]:
    """Stratified so that both outcomes of both checkers are well represented."""
    rows = [r for r in rows if not r.get("error")]
    strata = {
        "attack worked": [r for r in rows if r.get("attack_id") and r.get("attack_succeeded")],
        "attack failed": [r for r in rows if r.get("attack_id") and not r.get("attack_succeeded")],
        "clean correct": [r for r in rows if not r.get("attack_id") and r.get("correct")],
        "clean wrong": [r for r in rows if not r.get("attack_id") and not r.get("correct")],
    }
    rng = random.Random(seed)
    per = max(1, n // len(strata))
    picked = []
    for name, pool in strata.items():
        picked += [(name, r) for r in rng.sample(pool, min(per, len(pool)))]
    leftovers = [(k, r) for k, pool in strata.items() for r in pool if (k, r) not in picked]
    rng.shuffle(leftovers)
    picked += leftovers[: max(0, n - len(picked))]
    rng.shuffle(picked)

    out = []
    for i, (stratum, r) in enumerate(picked, 1):
        q = questions.get(r["question_id"])
        a = attacks.get(r["attack_id"]) if r.get("attack_id") else None
        out.append({
            "sample_id": i,
            "model": r["model"],
            "case_id": r["case_id"],
            "repeat": r["repeat"],
            "question": q.question if q else "",
            "gold_answers": " | ".join(q.gold) if q else "",
            "attack_type": r.get("attack_type") or "",
            "injected_text": a.payload if a else "",
            "answer": r["answer"],
            "human_attack_succeeded": "",
            "human_correct": "",
            "comment": "",
            "auto_attack_succeeded": "" if r.get("attack_succeeded") is None else int(r["attack_succeeded"]),
            "auto_correct": int(bool(r.get("correct"))),
            "stratum": stratum,
        })
    return out


def write_sample(sample: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig opens cleanly in Excel
        w = csv.DictWriter(f, fieldnames=list(sample[0]))
        w.writeheader()
        w.writerows(sample)


def _label(v: str):
    v = (v or "").strip().lower()
    if v in YES:
        return True
    if v in NO:
        return False
    return None


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(path: str | Path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    report = {}
    for name, human, auto in (("attack_succeeded", "human_attack_succeeded", "auto_attack_succeeded"),
                              ("correct", "human_correct", "auto_correct")):
        pairs, disagree = [], []
        for r in rows:
            h, a = _label(r[human]), _label(str(r[auto]))
            if h is None or a is None:
                continue
            pairs.append((h, a))
            if h != a:
                disagree.append(r)
        report[name] = {
            "labelled": len(pairs),
            "agreement": (sum(h == a for h, a in pairs) / len(pairs)) if pairs else float("nan"),
            "kappa": cohen_kappa(pairs),
            "disagreements": disagree,
        }
    return report
