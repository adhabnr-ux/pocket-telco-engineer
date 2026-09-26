"""Run every arm x seed of a study, then build the comparison figure.

    python experiments/run_matrix.py experiments/budget_scaling.yaml
    python experiments/run_matrix.py experiments/budget_scaling.yaml --dry-run

Each arm points at a run config; each seed gets its own run directory
(``<study_dir>/<arm>/seed<k>``). Finished runs are skipped, interrupted
runs resume, so you can stop and restart the study at any time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pte.config import load_config  # noqa: E402
from pte.llm import build_llm  # noqa: E402
from pte.report import compare_runs, run_report  # noqa: E402
from pte.search import Search  # noqa: E402
from pte.task import Task  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="run a single arm by name")
    args = ap.parse_args()

    study_path = Path(args.study).resolve()
    study = yaml.safe_load(study_path.read_text())
    out_dir = (study_path.parent / study.get("out_dir", "results/" + study_path.stem)).resolve()
    total_cap = float(study.get("total_budget_usd", 1e9))
    spent_total = 0.0
    run_dirs = []

    for arm in study["arms"]:
        if args.only and arm["name"] != args.only:
            continue
        for seed in arm.get("seeds", study.get("seeds", [0])):
            cfg = load_config((study_path.parent / arm["config"]).resolve())
            cfg.run_dir = str(out_dir / arm["name"] / f"seed{seed}")
            cfg.seed = seed
            for k, v in (arm.get("override", {}).get("search") or {}).items():
                setattr(cfg.search, k, v)
            for k, v in (arm.get("override", {}).get("budget") or {}).items():
                setattr(cfg.budget, k, v)
            cfg.worker.temperature = arm.get("temperature", cfg.worker.temperature)
            run_dirs.append(cfg.run_dir)
            final = Path(cfg.run_dir) / "final.json"
            if final.exists():
                spent_total += json.loads(final.read_text())["budget"]["spent_usd"]
                print(f"[skip] {arm['name']} seed {seed}: already finished")
                continue
            if spent_total + cfg.budget.max_usd > total_cap:
                print(f"[stop] next run could exceed the study cap of ${total_cap:.2f} (spent ${spent_total:.2f})")
                break
            print(f"[run ] {arm['name']} seed {seed}: {cfg.worker.provider}:{cfg.worker.model}, "
                  f"cap ${cfg.budget.max_usd:.2f} -> {cfg.run_dir}")
            if args.dry_run:
                continue
            task = Task(cfg.task)
            Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
            (Path(cfg.run_dir) / "config.json").write_text(json.dumps(
                {"task": cfg.task, "arm": arm["name"], "seed": seed}, indent=2))
            summary = Search(cfg, task, build_llm(cfg.orchestrator), build_llm(cfg.worker)).run()
            run_report(cfg.run_dir, task)
            spent_total += summary["budget"]["spent_usd"]

    done = [d for d in run_dirs if (Path(d) / "final.json").exists()]
    if done and not args.dry_run:
        task = Task(load_config((study_path.parent / study["arms"][0]["config"]).resolve()).task)
        fig = compare_runs(done, task, out_dir / "comparison.png")
        print(f"comparison: {fig} and {fig.with_suffix('.md')} | total spend ${spent_total:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
