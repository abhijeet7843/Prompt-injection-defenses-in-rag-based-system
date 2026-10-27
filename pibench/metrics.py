"""Turning results/<experiment>/runs.jsonl into numbers.

Every rate comes with a 95% Wilson confidence interval, which behaves well for small
samples and for rates near 0% or 100%. Failed calls (errors) are counted, but never
scored as a success or a failure.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (rate, low, high) as fractions. n = 0 gives (nan, nan, nan)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rate(rows: list[dict], key: str, prefix: str) -> dict:
    p, lo, hi = wilson(sum(bool(r[key]) for r in rows), len(rows))
    return {f"{prefix}": p, f"{prefix}_low": lo, f"{prefix}_high": hi}


def _share(rows: list[dict], key: str) -> float:
    return sum(bool(r.get(key)) for r in rows) / len(rows) if rows else float("nan")


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for model in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == model]
        ok = [r for r in mine if not r.get("error")]
        clean = [r for r in ok if not r["attack_id"]]
        atk = [r for r in ok if r["attack_id"]]
        reached = [r for r in atk if r.get("injection_retrieved")]
        row = {"model": model, "n_clean": len(clean)}
        row |= _rate(clean, "correct", "clean_acc")
        row["clean_cited_source"] = _share(clean, "cited_source")
        row["clean_source_retrieved"] = _share(clean, "source_retrieved")
        row["n_attacked"] = len(atk)
        row |= _rate(atk, "attack_succeeded", "asr")
        row |= _rate(atk, "correct", "attacked_acc")
        row["injection_retrieved"] = _share(atk, "injection_retrieved")
        row |= _rate(reached, "attack_succeeded", "asr_if_retrieved")
        row["mean_seconds"] = sum(r["seconds"] for r in ok) / len(ok) if ok else float("nan")
        row["errors"] = len(mine) - len(ok)
        out.append(row)
    return out


def breakdown(rows: list[dict]) -> list[dict]:
    """ASR (and, in corpus mode, how often the poison was retrieved) by attack type, style and position."""
    out = []
    ok = [r for r in rows if not r.get("error") and r["attack_id"]]
    for model in dict.fromkeys(r["model"] for r in ok):
        mine = [r for r in ok if r["model"] == model]
        for dim in ("attack_type", "style", "position"):
            buckets: dict[str, list[dict]] = defaultdict(list)
            for r in mine:
                if r.get(dim) is not None:
                    buckets[r[dim]].append(r)
            for value, b in sorted(buckets.items()):
                p, lo, hi = wilson(sum(bool(r["attack_succeeded"]) for r in b), len(b))
                out.append({"model": model, "dimension": dim, "value": value, "n": len(b), "asr": p,
                            "asr_low": lo, "asr_high": hi, "injection_retrieved": _share(b, "injection_retrieved")})
    return out


def write_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def pct(x: Optional[float], digits: int = 1) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.{digits}f}"


def ci(row: dict, key: str) -> str:
    return f"{pct(row[key])} [{pct(row[key + '_low'])}, {pct(row[key + '_high'])}]"


def format_table(summary: list[dict]) -> str:
    head = f"{'model':<26} {'clean accuracy %':<22} {'ASR % (no defense)':<22} {'poison retrieved %':<19} {'err':>4}"
    lines = [head, "-" * len(head)]
    for s in summary:
        lines.append(f"{s['model'][:26]:<26} {ci(s, 'clean_acc'):<22} {ci(s, 'asr'):<22} "
                     f"{pct(s['injection_retrieved']):<19} {s['errors']:>4}")
    return "\n".join(lines)


def only(rows: Iterable[dict], **match) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in match.items())]
