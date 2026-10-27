"""Running an experiment: every model x every case (x repeats), with caching and safe stopping."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from .attacks import build_cases, load_attack_files, load_questions, sample_cases
from .cache import ResponseCache
from .index import Index
from .models import build_model
from .pipeline import run_case
from .types import Case


def load_cases(cfg: dict[str, Any], sample: Optional[int] = None) -> list[Case]:
    attacks = load_attack_files(cfg["attacks"]) if cfg.get("attacks") else []
    cases = build_cases(load_questions(cfg["questions"]), attacks, include_clean=cfg["include_clean"])
    return sample_cases(cases, sample, seed=0) if sample else cases


def run_experiment(cfg: dict[str, Any], sample: Optional[int] = None, only_models: Optional[list[str]] = None,
                   verbose: bool = True) -> Path:
    cache = ResponseCache(cfg["cache_path"])
    index = Index.load_or_build(cfg, cache, verbose)
    cases = load_cases(cfg, sample)
    models = [build_model(m) for m in cfg["models"]]
    if only_models:
        models = [m for m in models if any(o in m.name for o in only_models)]
        if not models:
            raise SystemExit(f"no model in {cfg['_path']} matches {only_models}")

    out_dir = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    r = cfg["retrieval"]
    repeats = int(cfg["repeats"])
    total = len(models) * len(cases) * repeats
    done = errors = streak = 0
    t0 = time.time()
    stopped = False

    # runs.jsonl is rewritten on every run. The cache makes repeated runs cheap, and this keeps
    # the file free of rows produced by older code or data.
    with open(runs_path, "w", encoding="utf-8") as f:
        for model in models:
            for case in cases:
                for rep in range(repeats):
                    row, _ = run_case(case, model, index, secret=cfg["secret"], k=int(r["top_k"]), mode=r["mode"],
                                      injection_mode=cfg["injection_mode"], repeat=rep, cache=cache)
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
                    f.flush()
                    done += 1
                    if row.error:
                        errors += 1
                        streak += 1
                    else:
                        streak = 0
                    if verbose and (done % 50 == 0 or done == total or row.error):
                        eta = (total - done) * (time.time() - t0) / done / 60
                        msg = f"[{done}/{total}] {model.name} | {case.id}  ~{eta:.0f} min left"
                        if row.error:
                            msg += f"\n    ERROR: {row.error[:160]}"
                        print(msg, flush=True)
                    if streak >= int(cfg["max_consecutive_errors"]):
                        stopped = True
                        break
                if stopped:
                    break
            if stopped:
                break
    cache.close()
    if verbose:
        print(f"\n{done} of {total} runs written to {runs_path} in {(time.time() - t0) / 60:.1f} min, {errors} errors.")
        if stopped:
            print(f"STOPPED after {streak} errors in a row (often a used-up daily API quota or Ollama not "
                  "running). Fix the cause and run the same command again: finished calls come from the cache.")
    return runs_path
