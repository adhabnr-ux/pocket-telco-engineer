"""Build the main results table and figure from the discovered estimators.

    python scripts/results_table.py          # ~20-30 min on a laptop CPU

For every estimator: held-out test NVE on the batched track, on the single-slot track, and the
out-of-distribution NVE from results/stress_test.json (run scripts/stress_test.py first).
Writes results/main_table.{json,md} and figures/main_results.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pte.task import Task  # noqa: E402

D = REPO / "results" / "discovered"
B = REPO / "tasks" / "ofdm_channel_estimation" / "baselines"
ESTIMATORS = [  # (name, file, origin)
    ("ls_linear", B / "ls_linear.py", "reference"),
    ("dft_denoise", B / "dft_denoise.py", "textbook"),
    ("demo_hybrid", D / "demo_hybrid.py", "hand-made"),
    ("batched_agent1", D / "batched_agent1.py", "agent, batched track"),
    ("batched_agent2", D / "batched_agent2.py", "agent, batched track"),
    ("batched_agent3", D / "batched_agent3.py", "agent, batched track"),
    ("single_agent1", D / "single_agent1.py", "agent, single-slot track"),
    ("single_agent2", D / "single_agent2.py", "agent, single-slot track"),
    ("single_agent3", D / "single_agent3.py", "agent, single-slot track"),
    ("single_agent1_ablation_no_energy", D / "ablation_single_agent1_no_energy.py", "ablation (1 line removed)"),
]


def main() -> int:
    if "--plot-only" in sys.argv:
        plot(json.loads((REPO / "results" / "main_table.json").read_text()))
        return 0
    batched = Task(REPO / "tasks" / "ofdm_channel_estimation")
    single = Task(REPO / "tasks" / "ofdm_channel_estimation_single_slot")
    genie = batched.baselines()["test"]
    ref = genie["ls_linear"]
    genie_nve = float(np.mean([genie["lmmse_genie"][k] / ref[k] for k in ref]))
    stress = {r["estimator"]: r for r in json.loads((REPO / "results" / "stress_test.json").read_text())["rows"]}

    rows = []
    for name, path, origin in ESTIMATORS:
        rb = batched.evaluate_file(path, split="test", timeout_s=3600)
        rs = single.evaluate_file(path, split="test", timeout_s=3600, extra_args=["--no-timing"])
        rows.append({"estimator": name, "origin": origin,
                     "batched_test_nve": rb.metric if rb.success else None,
                     "single_slot_test_nve": rs.metric if rs.success else None,
                     "ood_nve": stress.get(name, {}).get("nve_ood"),
                     "runtime_ms_batch32": rb.complexity * 1e3 if rb.success else None})
        print(rows[-1], flush=True)
    rows.append({"estimator": "lmmse_genie", "origin": "not allowed: knows true statistics (pilot-only)",
                 "batched_test_nve": genie_nve, "single_slot_test_nve": genie_nve,
                 "ood_nve": stress.get("lmmse_genie", {}).get("nve_ood"), "runtime_ms_batch32": None})

    out = REPO / "results"
    (out / "main_table.json").write_text(json.dumps(rows, indent=1))
    f = lambda v, d=4: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    lines = ["| estimator | origin | batched track (test) | single-slot track (test) | out-of-distribution | runtime [ms] |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r['estimator']} | {r['origin']} | {f(r['batched_test_nve'])} | {f(r['single_slot_test_nve'])} | "
              f"{f(r['ood_nve'])} | {f(r['runtime_ms_batch32'], 1)} |" for r in rows]
    (out / "main_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(rows)
    return 0


def plot(rows) -> None:
    """Figure: three NVEs per estimator, log scale."""
    show = [r for r in rows if r["estimator"] not in ("dft_denoise",)]
    x = np.arange(len(show))
    w = 0.27
    fig, ax = plt.subplots(figsize=(12, 4.8))
    cols = {"batched_test_nve": ("#94a3b8", "batched track (pooling allowed)"),
            "single_slot_test_nve": ("#2563eb", "single-slot track (realistic)"),
            "ood_nve": ("#dc2626", "out-of-distribution channels")}
    for j, (k, (c, lab)) in enumerate(cols.items()):
        vals = [r[k] if r[k] is not None else np.nan for r in show]
        ax.bar(x + (j - 1) * w, vals, w, color=c, label=lab)
    ax.axhline(1.0, color="0.3", ls="--", lw=1)
    ax.text(len(show) - 0.5, 1.05, "textbook reference = 1.0", ha="right", fontsize=8, color="0.3")
    ax.set_yscale("log")
    ax.set_ylim(0.015, 12)
    ax.set_xticks(x, [r["estimator"].replace("_ablation_no_energy", "\n(minus 1 line)") for r in show],
                  rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("NVE: error relative to textbook (log, lower is better)")
    ax.set_title("AI-discovered channel estimators: benchmark score vs. realistic and unseen conditions")
    ax.legend(fontsize=8, loc="upper center", ncol=3)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    (REPO / "figures").mkdir(exist_ok=True)
    fig.savefig(REPO / "figures" / "main_results.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
