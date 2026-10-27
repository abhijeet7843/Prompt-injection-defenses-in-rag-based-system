"""Loading questions and attacks, building test cases, and placing the poisoned document.

Two threat models (set `injection_mode` in the config):

  forced  The retriever runs normally, then the poisoned chunk is inserted into the
          retrieved context at the attack's position (start, middle or end). The model
          always sees the attack, which isolates how the model (and later, a defense)
          reacts to it. This is the controlled setting used to compare defenses.

  corpus  The poisoned document is added to the searchable corpus and must win a place in
          the top k on its own, as in a real knowledge-base poisoning attack. This measures
          the end-to-end risk, including how often retrieval surfaces the attack at all.
"""
from __future__ import annotations

import json
from pathlib import Path

from .types import Attack, Case, Chunk, Question


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}, line {n}: not valid JSON ({e})") from e
    return rows


def load_questions(path: str | Path) -> dict[str, Question]:
    qs = {}
    for r in _read_jsonl(path):
        q = Question(id=r["id"], question=r["question"], gold=r["gold"], source=r.get("source", ""))
        if q.id in qs:
            raise ValueError(f"Duplicate question id {q.id}")
        qs[q.id] = q
    return qs


def load_attacks(path: str | Path) -> list[Attack]:
    attacks, seen = [], set()
    for r in _read_jsonl(path):
        unknown = set(r) - set(Attack.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path}: attack {r.get('id')} has unknown fields {sorted(unknown)}")
        a = Attack(**r)
        if a.id in seen:
            raise ValueError(f"Duplicate attack id {a.id}")
        if a.position not in {"start", "middle", "end"}:
            raise ValueError(f"Attack {a.id}: position must be start, middle or end")
        if a.success_kind not in {"marker", "leak", "tool_call"}:
            raise ValueError(f"Attack {a.id}: success_kind must be marker, leak or tool_call")
        seen.add(a.id)
        attacks.append(a)
    return attacks


def load_attack_files(paths) -> list[Attack]:
    """`attacks:` in a config can be one file or a list of files."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out = []
    for p in paths:
        out += load_attacks(p)
    ids = [a.id for a in out]
    if len(ids) != len(set(ids)):
        raise ValueError("the same attack id appears in more than one attack file")
    return out


def build_cases(questions: dict[str, Question], attacks: list[Attack],
                include_clean: bool = True) -> list[Case]:
    cases = []
    if include_clean:
        cases += [Case(id=f"clean:{q.id}", question=q) for q in questions.values()]
    for a in attacks:
        if a.question_id not in questions:
            raise ValueError(f"Attack {a.id} points to unknown question {a.question_id}")
        cases.append(Case(id=f"atk:{a.id}", question=questions[a.question_id], attack=a))
    return cases


def poisoned_chunk(attack: Attack) -> Chunk:
    """The attacker's document. Its carrier line poses as the page title, like a real look-alike page."""
    return Chunk(doc_id=f"injected:{attack.id}#0", title=attack.carrier.rstrip("."), text=attack.payload,
                 injected=True)


def sample_cases(cases: list[Case], n: int, seed: int = 0) -> list[Case]:
    """A reproducible random subset that keeps the clean/attacked mix, for pilot runs."""
    import random

    if n >= len(cases):
        return list(cases)
    rng = random.Random(seed)
    clean = [c for c in cases if not c.is_attacked]
    atk = [c for c in cases if c.is_attacked]
    n_clean = round(n * len(clean) / len(cases))
    picked = rng.sample(clean, min(n_clean, len(clean))) + rng.sample(atk, min(n - n_clean, len(atk)))
    order = {c.id: i for i, c in enumerate(cases)}
    return sorted(picked, key=lambda c: order[c.id])


def inject(chunks: list[Chunk], attack: Attack) -> list[Chunk]:
    """Return a new chunk list with the attacker's chunk inserted."""
    poisoned = poisoned_chunk(attack)
    out = list(chunks)
    if attack.position == "start":
        out.insert(0, poisoned)
    elif attack.position == "end":
        out.append(poisoned)
    else:
        out.insert(len(out) // 2 if out else 0, poisoned)
    return out
