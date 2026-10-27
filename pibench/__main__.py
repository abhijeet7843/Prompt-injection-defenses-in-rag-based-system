"""Command line. Run from the repo folder; every command takes --config (default configs/baseline.yaml).

  python -m pibench check                    can each model, and the embedder, be reached?
  python -m pibench index                    build the search index (rebuilt only if something changed)
  python -m pibench retrieval                recall@k and MRR for BM25, dense and hybrid search
  python -m pibench ask "When is the fee deadline?"          one question, every step printed
  python -m pibench ask x --attack s_q01_override            one attacked question
  python -m pibench plan                     how many model calls an experiment makes
  python -m pibench run --sample 40          a quick pilot on a random subset
  python -m pibench run                      the whole experiment
  python -m pibench report                   tables and figures
  python -m pibench sample --n 60            a spreadsheet of outputs to check by hand
  python -m pibench agreement                how often the automatic checkers agree with you
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .attacks import load_attack_files, load_questions
from .cache import ResponseCache
from .config import load_config, load_env, read_env_text
from .embeddings import EmbeddingError, build_embedder
from .handcheck import agreement, make_sample, write_sample
from .index import MODES, Index
from .metrics import load_rows, pct, write_csv
from .models import build_model, list_gemini_models
from .pipeline import run_case
from .report import build_report
from .retrieval_eval import evaluate, format_table
from .runner import load_cases, run_experiment
from .types import Case, Messages, Question


def _open(cfg):
    cache = ResponseCache(cfg["cache_path"])
    try:
        return cache, Index.load_or_build(cfg, cache)
    except EmbeddingError as e:
        sys.exit(f"Could not embed the corpus: {e}")


def _pick_model(cfg, name):
    if name == "mock":
        return build_model({"provider": "mock"})
    specs = cfg["models"]
    if name:
        specs = [s for s in specs if name in f"{s['provider']}:{s.get('model', '')}"]
    if not specs:
        sys.exit(f"No model matching '{name}' in {cfg['_path']}")
    return build_model(specs[0])


def _env_report() -> str:
    """What .env contains, by variable name only. Never prints a value."""
    p = Path(".env")
    if not p.exists():
        return f"no .env file in {Path.cwd()}"
    lines = [ln.strip() for ln in read_env_text(p).splitlines() if ln.strip() and not ln.strip().startswith("#")]
    found = []
    for ln in lines:
        name, sep, value = ln.partition("=")
        found.append(f"{name.strip()!r} ({'has a value' if value.strip() else 'EMPTY'})" if sep else
                     f"a line without '=' ({len(ln)} characters)")
    return ".env holds: " + (", ".join(found) if found else "nothing (the key line is missing)")


def cmd_check(args, cfg):
    ok = True
    probe = Messages(system="Reply with one word.", user="Say OK.")
    for spec in cfg["models"]:
        m = build_model(spec)
        print(f"...   asking {m.name} (a local model can take up to a minute the first time)", flush=True)
        r = m.generate(probe)
        ok &= r.error is None
        print(f"FAIL  {m.name}: {r.error[:220]}" if r.error else f"OK    {m.name}: replied {r.text[:40]!r} in {r.seconds:.1f}s",
              flush=True)
        if r.error and "GEMINI_API_KEY" in r.error:
            print(f"      {_env_report()}. The line must read GEMINI_API_KEY=your-key", flush=True)
    emb = build_embedder(cfg.get("embedder"))
    if emb is not None:
        try:
            v = emb.embed(["The library opens at 8 am."], "document")
            print(f"OK    embedder {emb.name}: {v.shape[1]} dimensions")
        except EmbeddingError as e:
            ok = False
            print(f"FAIL  embedder {emb.name}: {e}")
    sys.exit(0 if ok else 1)


def cmd_gemini_models(args, cfg):
    ids, err = list_gemini_models()
    if err:
        sys.exit(f"Could not list Gemini models: {err}")
    flash = [i for i in ids if "flash" in i and "image" not in i and "tts" not in i and "audio" not in i]
    print("Gemini models your key can use for text (Flash family):")
    for i in sorted(flash):
        print("  " + i)
    print(f"\n({len(ids)} models in total. Free-tier daily limits per model are shown at aistudio.google.com/rate-limit)")


def cmd_index(args, cfg):
    if args.rebuild:
        Path(cfg["index_dir"], "manifest.json").unlink(missing_ok=True)
    cache, idx = _open(cfg)
    print(json.dumps(idx.info(), indent=2))
    cache.close()


def cmd_retrieval(args, cfg):
    cache, idx = _open(cfg)
    summary, per_q = evaluate(idx, load_questions(cfg["questions"]))
    out = Path(cfg["results_dir"]).parent / "retrieval"
    write_csv(summary, out / "retrieval_summary.csv")
    write_csv(per_q, out / "retrieval_per_question.csv")
    print(f"{idx.info()['n_chunks']} chunks from {idx.info()['n_documents']} documents, embedder: "
          f"{idx.info()['embedder']}\n")
    print(format_table(summary))
    misses = [r for r in per_q if (r["rank"] or 99) > 3]
    if misses:
        print("\nQuestions whose answer passage is not in the top 3:")
        for r in misses:
            print(f"  {r['mode']:<7} {r['question_id']}  rank {r['rank'] or '>10'}")
    print(f"\nSaved {out}/retrieval_summary.csv and retrieval_per_question.csv")
    cache.close()


def cmd_ask(args, cfg):
    cache, idx = _open(cfg)
    questions = load_questions(cfg["questions"])
    attack = None
    if args.attack:
        attack = next((a for a in load_attack_files(cfg.get("attacks") or "data/attacks.jsonl")
                       if a.id == args.attack), None)
        if attack is None:
            sys.exit(f"No attack with id {args.attack}")
        q = questions[attack.question_id]
    else:
        q = next((x for x in questions.values() if x.question.strip().lower() == args.question.strip().lower()),
                 Question(id="adhoc", question=args.question, gold=[]))
    model = _pick_model(cfg, args.model)
    r = cfg["retrieval"]
    row, messages = run_case(Case("ask", q, attack), model, idx, secret=cfg["secret"], k=r["top_k"],
                             mode=args.retrieval or r["mode"], injection_mode=args.mode or cfg["injection_mode"],
                             cache=None if args.no_cache else cache)
    print(f"=== model: {model.name} | retrieval: {row.retrieval} | attack: {args.attack or 'none'}"
          f"{' (' + row.injection_mode + ')' if attack else ''} ===\n")
    if args.show_prompt:
        print("--- SYSTEM PROMPT ---\n" + messages.system + "\n")
    print("--- PROMPT ---\n" + messages.user + "\n")
    print("--- ANSWER ---\n" + (row.answer or "(empty)") + "\n")
    if row.error:
        print("ERROR:", row.error)
    print(f"Retrieved: {row.retrieved}")
    if q.gold:
        print(f"Correct: {row.correct}   cited the right document: {row.cited_source}")
    if row.tool_calls:
        print("Tool calls:", row.tool_calls)
    if attack:
        print(f"Poisoned chunk shown to the model: {row.injection_retrieved} (rank {row.injection_rank})")
        print(f"Attack succeeded: {row.attack_succeeded}")
    cache.close()


def cmd_plan(args, cfg):
    cases = load_cases(cfg, args.sample)
    n_models = len(cfg["models"])
    per_model = len(cases) * int(cfg["repeats"])
    n_atk = sum(c.is_attacked for c in cases)
    print(f"{len(cases)} cases ({len(cases) - n_atk} clean, {n_atk} attacked) x {cfg['repeats']} repeat(s) "
          f"= {per_model} calls per model, {per_model * n_models} for {n_models} model(s)")
    for secs in (1, 3, 6):
        print(f"  at {secs} s per call: about {per_model * secs / 60:.0f} minutes per model")
    print("Cached calls are free, so reruns only pay for new prompts.")


def cmd_run(args, cfg):
    try:
        run_experiment(cfg, sample=args.sample, only_models=args.models)
    except EmbeddingError as e:
        sys.exit(f"Could not embed: {e}")
    build_report(cfg["results_dir"], title=Path(cfg["_path"]).stem, figures=False)


def cmd_report(args, cfg):
    build_report(cfg["results_dir"], title=Path(cfg["_path"]).stem)


def cmd_sample(args, cfg):
    rows = load_rows(Path(cfg["results_dir"]) / "runs.jsonl")
    if not rows:
        sys.exit("No results yet: run the experiment first.")
    attacks = {a.id: a for a in load_attack_files(cfg["attacks"])} if cfg.get("attacks") else {}
    sample = make_sample(rows, load_questions(cfg["questions"]), attacks, n=args.n)
    path = Path(cfg["results_dir"]) / "handcheck.csv"
    if path.exists() and not args.overwrite:
        sys.exit(f"{path} already exists (it may hold your labels). Use --overwrite to replace it.")
    write_sample(sample, path)
    print(f"Wrote {len(sample)} rows to {path}. Fill the human_ columns with y or n, then run "
          f"`python -m pibench agreement --config {cfg['_path']}`.")


def cmd_agreement(args, cfg):
    path = Path(cfg["results_dir"]) / "handcheck.csv"
    if not path.exists():
        sys.exit(f"{path} not found. Run `python -m pibench sample` first.")
    for name, r in agreement(path).items():
        if not r["labelled"]:
            print(f"{name:<18} nothing labelled yet")
            continue
        print(f"{name:<18} labelled {r['labelled']:>3}   agreement {pct(r['agreement'])}%   "
              f"Cohen's kappa {r['kappa']:.2f}")
        for d in r["disagreements"]:
            print(f"    sample {d['sample_id']}: {d['model']} | {d['case_id']} -> {d['answer'][:100]!r}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="pibench", description="Prompt-injection benchmark: week-4 baseline")
    p.add_argument("--config", default="configs/baseline.yaml")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)  # also accepted after the command name
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, help_):
        return sub.add_parser(name, help=help_, parents=[common])

    add("check", "test that every model and the embedder respond")
    add("gemini-models", "list the Gemini models your API key can use")
    ix = add("index", "build the search index")
    ix.add_argument("--rebuild", action="store_true", help="rebuild even if nothing changed")
    add("retrieval", "evaluate BM25, dense and hybrid retrieval")
    a = add("ask", "run one question and print every step")
    a.add_argument("question")
    a.add_argument("--model", help="part of a model name from the config, or 'mock'")
    a.add_argument("--attack", help="attack id, e.g. s_q01_override")
    a.add_argument("--mode", choices=["forced", "corpus"], help="how the attack is placed")
    a.add_argument("--retrieval", choices=list(MODES))
    a.add_argument("--show-prompt", action="store_true", help="also print the system prompt")
    a.add_argument("--no-cache", action="store_true", help="always call the model")
    pl = add("plan", "count the model calls an experiment makes")
    pl.add_argument("--sample", type=int)
    r = add("run", "run the experiment in the config")
    r.add_argument("--sample", type=int, help="only a random subset of N cases (a pilot)")
    r.add_argument("--models", nargs="+", help="only models whose name contains these words")
    add("report", "tables and figures for the experiment")
    s = add("sample", "write a spreadsheet of outputs to check by hand")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--overwrite", action="store_true")
    add("agreement", "compare your hand labels with the automatic checkers")

    args = p.parse_args(argv)
    load_env()
    cfg = load_config(args.config)
    {"check": cmd_check, "gemini-models": cmd_gemini_models, "index": cmd_index, "retrieval": cmd_retrieval, "ask": cmd_ask, "plan": cmd_plan,
     "run": cmd_run, "report": cmd_report, "sample": cmd_sample, "agreement": cmd_agreement}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
