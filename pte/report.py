"""Figures and a Markdown report for one run, and a comparison across runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .leaderboard import Leaderboard
from .task import Task


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def baseline_nves(task: Task, split: str) -> dict[str, float]:
    b = task.baselines()
    if split not in b:
        return {}
    ref = b[split]["ls_linear"]
    return {name: float(np.mean([vals[k] / ref[k] for k in ref])) for name, vals in b[split].items()}


def _baseline_lines(ax, base: dict[str, float], axis: str = "y"):
    styles = {"ls_nearest": ":", "ls_linear": "--", "dft_denoise": "-.", "lmmse_genie": (0, (1, 1))}
    for name, v in base.items():
        if axis == "y":
            ax.axhline(v, ls=styles.get(name, "--"), lw=1, color="0.45")
            ax.annotate(name, (1.0, v), xycoords=("axes fraction", "data"), xytext=(-4, 2),
                        textcoords="offset points", ha="right", va="bottom", fontsize=7, color="0.35")


def run_report(run_dir: str | Path, task: Task) -> Path:
    run_dir = Path(run_dir)
    board = Leaderboard(run_dir / "leaderboard.jsonl", task.higher_is_better)
    final = json.loads((run_dir / "final.json").read_text()) if (run_dir / "final.json").exists() else {}
    base_dev = baseline_nves(task, "dev")
    base_test = baseline_nves(task, "test")
    plt = _plt()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # (1) best-so-far vs spend
    curve = board.best_so_far_curve()
    ax = axes[0]
    if curve:
        spend = [c[0] for c in curve]
        calls = [c[1] for c in curve]
        best = [c[3] for c in curve]
        x = spend if max(spend) > 0 else calls
        ax.step(x, best, where="post", color="#2563eb", lw=2, label="best candidate (dev)")
        ax.set_xlabel("cumulative LLM spend [USD]" if max(spend) > 0 else "cumulative LLM calls")
    _baseline_lines(ax, base_dev)
    ax.set_ylabel("NVE (lower is better)")
    ax.set_title("Discovery progress vs. budget")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

    # (2) all candidates: metric vs complexity with Pareto front
    ax = axes[1]
    ok = board.successes()
    if ok:
        ax.scatter([c.complexity * 1e3 for c in ok], [c.metric for c in ok], s=14, alpha=0.5,
                   color="#64748b", label="candidates")
        front = board.pareto_front()
        ax.plot([c.complexity * 1e3 for c in front], [c.metric for c in front], "o-", color="#dc2626",
                ms=5, lw=1.5, label="Pareto front")
        ax.legend(fontsize=8, loc="lower left")
    _baseline_lines(ax, base_dev)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("runtime per call, batch 32 [ms]")
    ax.set_ylabel("NVE (lower is better)")
    ax.set_title("Accuracy vs. complexity")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig_path = run_dir / "report.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    n_fail = len(board.items) - len(ok)
    lines = [
        f"# Run report: `{run_dir.name}`",
        "",
        f"- Task: `{task.name}`",
        f"- Worker model: `{final.get('worker_model', '?')}`, orchestrator: `{final.get('orchestrator_model', '?')}`",
        f"- Spend: ${final.get('budget', {}).get('spent_usd', 0):.4f} of ${final.get('max_usd', 0):.2f} "
        f"in {final.get('budget', {}).get('calls', 0)} LLM calls, {final.get('elapsed_s', 0) / 60:.1f} min",
        f"- Candidates: {len(board.items)} evaluated, {len(ok)} succeeded, {n_fail} failed",
        f"- Stopped because: {final.get('stopped_reason') or 'search completed'}",
        "",
        "![report](report.png)",
        "",
        "## Reference estimators",
        "",
        "| estimator | dev NVE | held-out test NVE |",
        "|---|---|---|",
    ]
    for name in base_dev:
        lines.append(f"| {name} | {base_dev[name]:.4f} | {base_test.get(name, float('nan')):.4f} |")
    lines += ["", "## Best discovered candidates (held-out re-evaluation)", "",
              "| candidate | idea | dev NVE | test NVE | runtime [ms] |", "|---|---|---|---|---|"]
    for f in final.get("final", []):
        c = f["candidate"]
        t = f"{f['test_metric']:.4f}" if f.get("test_success") else "FAILED"
        lines.append(f"| `{c['id']}` | {c['idea']} | {f['dev_metric']:.4f} | {t} | {c['complexity'] * 1e3:.2f} |")
    front = board.pareto_front()
    if front:
        lines += ["", "## Pareto front (dev)", "", "| candidate | NVE | runtime [ms] |", "|---|---|---|"]
        lines += [f"| `{c.id}` | {c.metric:.4f} | {c.complexity * 1e3:.2f} |" for c in front]
    md = run_dir / "REPORT.md"
    md.write_text("\n".join(lines) + "\n")
    return md


def compare_runs(run_dirs: list[str | Path], task: Task, out: str | Path) -> Path:
    """Overlay best-so-far-vs-spend curves of several runs (e.g. different models / seeds)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    for rd in run_dirs:
        rd = Path(rd)
        board = Leaderboard(rd / "leaderboard.jsonl", task.higher_is_better)
        curve = board.best_so_far_curve()
        if not curve:
            continue
        final = json.loads((rd / "final.json").read_text()) if (rd / "final.json").exists() else {}
        label = f"{rd.name} ({final.get('worker_model', '?')})"
        ax.step([c[0] for c in curve], [c[3] for c in curve], where="post", lw=1.8, label=label)
        tests = [f["test_metric"] for f in final.get("final", []) if f.get("test_success")]
        rows.append((rd.name, final.get("worker_model", "?"), curve[-1][3], min(tests) if tests else None,
                     final.get("budget", {}).get("spent_usd", curve[-1][0])))
    _baseline_lines(ax, baseline_nves(task, "dev"))
    ax.set_xlabel("cumulative LLM spend [USD]")
    ax.set_ylabel("best NVE so far (dev, lower is better)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax.set_title("Algorithm discovery vs. budget")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    table = ["| run | worker model | best dev NVE | best test NVE | spend [USD] |", "|---|---|---|---|---|"]
    for r in rows:
        t = f"{r[3]:.4f}" if r[3] is not None else "-"
        table.append(f"| {r[0]} | {r[1]} | {r[2]:.4f} | {t} | {r[4]:.4f} |")
    out.with_suffix(".md").write_text("\n".join(table) + "\n")
    return out
