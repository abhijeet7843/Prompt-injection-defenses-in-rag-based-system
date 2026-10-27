"""Tables and figures for one experiment, built from its runs.jsonl.

    python -m pibench report --config configs/attacks_forced.yaml

Writes into <results_dir>/: summary.csv, breakdown.csv, results.md, and figures/:
  asr_by_type_style.png   attack success by attack type, one bar per style, one panel per model
  asr_by_position.png     attack success by where the poisoned chunk sits (forced mode)
"""
from __future__ import annotations

from pathlib import Path

from .metrics import breakdown, ci, format_table, load_rows, pct, summarize, write_csv

# Reference palette (categorical slots 1-5, checked for colour-vision deficiency). Text never uses series colours.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TYPES = ["override", "hijack", "leak", "tool_misuse"]
STYLES = ["plain", "authority", "fake_dialogue", "embedded_note", "markup"]
POSITIONS = ["start", "middle", "end"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "savefig.dpi": 200, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt, FuncFormatter(lambda v, _: f"{v:.0f}%")


def _bars(ax, groups, series, value, err=None, labels=None):
    import numpy as np

    width = 0.8 / len(series)
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        vals = [value(g, s) for g in groups]
        kw = {}
        if err:
            lo_hi = [err(g, s) for g in groups]
            kw = dict(yerr=[[max(0, v - lo) for v, (lo, _) in zip(vals, lo_hi)],
                            [max(0, hi - v) for v, (_, hi) in zip(vals, lo_hi)]],
                      error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=0))
        ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width, color=SERIES[i % len(SERIES)],
               edgecolor="white", linewidth=1, label=(labels or {}).get(s, s.replace("_", " ")), **kw)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", " ") for g in groups])
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)


def fig_type_style(rows: list[dict], out: Path) -> None:
    atk = [r for r in rows if r["attack_id"] and not r.get("error")]
    if not atk:
        return
    plt, fmt = _plt()
    models = list(dict.fromkeys(r["model"] for r in atk))
    types = [t for t in TYPES if any(r["attack_type"] == t for r in atk)]
    styles = [s for s in STYLES if any(r["style"] == s for r in atk)]

    def rate(model, t, s):
        b = [r for r in atk if r["model"] == model and r["attack_type"] == t and r["style"] == s]
        return 100 * sum(bool(r["attack_succeeded"]) for r in b) / len(b) if b else float("nan")

    fig, axes = plt.subplots(len(models), 1, figsize=(7.2, 2.6 * len(models) + 0.4), squeeze=False)
    for ax, model in zip(axes[:, 0], models):
        _bars(ax, types, styles, lambda t, s, m=model: rate(m, t, s))
        ax.yaxis.set_major_formatter(fmt)
        ax.set_ylabel("Attack success rate")
        ax.set_title(model, fontsize=9.5, loc="left")
    axes[0, 0].legend(ncol=len(styles), fontsize=7.5, loc="lower right", bbox_to_anchor=(1, 1.08))
    fig.suptitle("Attack success with no defense, by attack type and style", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_position(bd: list[dict], out: Path) -> None:
    pos = [r for r in bd if r["dimension"] == "position"]
    if not pos:
        return
    plt, fmt = _plt()
    models = list(dict.fromkeys(r["model"] for r in pos))
    cell = {(r["model"], r["value"]): r for r in pos}
    fig, ax = plt.subplots(figsize=(max(4.5, 1.6 * len(models) + 1.5), 3.0))
    _bars(ax, models, POSITIONS,
          lambda m, p: 100 * cell[(m, p)]["asr"] if (m, p) in cell else float("nan"),
          lambda m, p: ((100 * cell[(m, p)]["asr_low"], 100 * cell[(m, p)]["asr_high"]) if (m, p) in cell
                        else (float("nan"),) * 2),
          labels={p: f"poison at {p}" for p in POSITIONS})
    ax.set_xticklabels(models, fontsize=8)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_ylabel("Attack success rate")
    ax.legend(ncol=3, fontsize=7.5, loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.set_title("By position of the poisoned chunk (95% intervals)", fontsize=9.5, loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def results_markdown(summary: list[dict], bd: list[dict], title: str) -> str:
    md = [f"# Results: {title}", "",
          "Percentages with 95% Wilson intervals. ASR = attack success rate. No defense is active. "
          "Errors are failed calls, excluded from every rate.", ""]
    md += ["## Clean questions", "",
           "| Model | n | Accuracy | Cited the right document | Right document retrieved |", "|---|---|---|---|---|"]
    for s in summary:
        if s["n_clean"]:
            md.append(f"| {s['model']} | {s['n_clean']} | {ci(s, 'clean_acc')} | {pct(s['clean_cited_source'])} | "
                      f"{pct(s['clean_source_retrieved'])} |")
    if any(s["n_attacked"] for s in summary):
        md += ["", "## Attacked questions", "",
               "| Model | n | ASR | Accuracy under attack | Poison retrieved | ASR when retrieved | Errors |",
               "|---|---|---|---|---|---|---|"]
        for s in summary:
            if s["n_attacked"]:
                md.append(f"| {s['model']} | {s['n_attacked']} | {ci(s, 'asr')} | {ci(s, 'attacked_acc')} | "
                          f"{pct(s['injection_retrieved'])} | {ci(s, 'asr_if_retrieved')} | {s['errors']} |")
    for dim, label in (("attack_type", "attack type"), ("style", "style"), ("position", "position")):
        rows = [r for r in bd if r["dimension"] == dim]
        if not rows:
            continue
        values = list(dict.fromkeys(r["value"] for r in rows))
        md += ["", f"## ASR by {label} (%)", "", "| Model | " + " | ".join(values) + " |",
               "|---|" + "---|" * len(values)]
        for model in dict.fromkeys(r["model"] for r in rows):
            cells = {r["value"]: r for r in rows if r["model"] == model}
            md.append(f"| {model} | " + " | ".join(
                f"{pct(cells[v]['asr'])} (n={cells[v]['n']})" if v in cells else "" for v in values) + " |")
    return "\n".join(md) + "\n"


def build_report(results_dir: str | Path, title: str = "", figures: bool = True) -> list[dict]:
    out = Path(results_dir)
    rows = load_rows(out / "runs.jsonl")
    if not rows:
        raise SystemExit(f"No results in {out}/runs.jsonl yet. Run the experiment first.")
    summary, bd = summarize(rows), breakdown(rows)
    write_csv(summary, out / "summary.csv")
    write_csv(bd, out / "breakdown.csv")
    (out / "results.md").write_text(results_markdown(summary, bd, title or out.name), encoding="utf-8")
    if figures:
        (out / "figures").mkdir(exist_ok=True)
        fig_type_style(rows, out / "figures/asr_by_type_style.png")
        fig_position(bd, out / "figures/asr_by_position.png")
    print(format_table(summary))
    print(f"\nWrote summary.csv, breakdown.csv, results.md and figures to {out}/")
    return summary
