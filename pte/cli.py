"""Command line interface: ``pte run | demo | eval | report | compare | check-llm``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .config import RunConfig, load_config
from .llm import Message, build_llm
from .task import Task

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TASK = REPO / "tasks" / "ofdm_channel_estimation"


def _save_config(cfg: RunConfig) -> None:
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))


def _task_for_run(run_dir: Path, override: str | None) -> Task:
    if override:
        return Task(override)
    cfg = run_dir / "config.json"
    if cfg.exists():
        return Task(json.loads(cfg.read_text())["task"])
    return Task(DEFAULT_TASK)


def cmd_run(args) -> int:
    from .report import run_report
    from .search import Search

    cfg = load_config(args.config)
    if args.run_dir:
        cfg.run_dir = str(Path(args.run_dir).resolve())
    task = Task(cfg.task)
    _save_config(cfg)
    print(f"task: {task.name} | worker: {cfg.worker.model} | orchestrator: {cfg.orchestrator.model} "
          f"| budget: ${cfg.budget.max_usd:.2f} | run dir: {cfg.run_dir}")
    search = Search(cfg, task, build_llm(cfg.orchestrator), build_llm(cfg.worker))
    try:
        search.run()
    except KeyboardInterrupt:
        print("\ninterrupted: progress is saved; re-run the same command to resume.")
        search._save_state()
        return 130
    md = run_report(cfg.run_dir, task)
    print(f"report: {md}")
    return 0


def cmd_demo(args) -> int:
    from .config import BudgetConfig, EvalConfig, SearchConfig
    from .demo import scripted_llms
    from .llm import ModelSpec
    from .report import run_report
    from .search import Search

    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        run_dir = run_dir.with_name(run_dir.name + time.strftime("_%Y%m%d_%H%M%S"))
    cfg = RunConfig(
        task=str(DEFAULT_TASK), run_dir=str(run_dir),
        orchestrator=ModelSpec(provider="scripted", model="scripted-orchestrator"),
        worker=ModelSpec(provider="scripted", model="scripted-worker"),
        search=SearchConfig(generations=1, ideas_per_generation=2, workers_per_generation=2,
                            steps_per_worker=4, max_parallel=2),
        budget=BudgetConfig(max_usd=1.0, max_calls=100),
        eval=EvalConfig(final_top_k=3),
        notes="offline demo with a scripted LLM",
    )
    task = Task(cfg.task)
    _save_config(cfg)
    orch, worker = scripted_llms(DEFAULT_TASK)
    Search(cfg, task, orch, worker).run()
    md = run_report(run_dir, task)
    print(f"\ndemo finished. report: {md}")
    return 0


def cmd_eval(args) -> int:
    task = Task(args.task)
    res = task.evaluate_file(Path(args.file), split=args.split, timeout_s=args.timeout)
    print(res.output.rstrip())
    return 0 if res.success else 1


def cmd_report(args) -> int:
    from .report import run_report

    run_dir = Path(args.run_dir)
    print(run_report(run_dir, _task_for_run(run_dir, args.task)))
    return 0


def cmd_compare(args) -> int:
    from .report import compare_runs

    task = _task_for_run(Path(args.run_dirs[0]), args.task)
    out = compare_runs(args.run_dirs, task, args.out)
    print(f"{out}\n{out.with_suffix('.md')}")
    return 0


def cmd_check_llm(args) -> int:
    cfg = load_config(args.config)
    for role, spec in (("worker", cfg.worker), ("orchestrator", cfg.orchestrator)):
        llm = build_llm(spec)
        t0 = time.time()
        reply = llm.complete("You are terse.", [Message("user", "Reply with exactly: OK")])
        cost = (reply.input_tokens * spec.price_in_per_mtok + reply.output_tokens * spec.price_out_per_mtok) / 1e6
        print(f"{role:<12} {spec.provider}:{spec.model} -> {reply.text.strip()[:40]!r} "
              f"({reply.input_tokens}+{reply.output_tokens} tokens, ${cost:.6f}, {time.time() - t0:.1f}s)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pte", description="pocket-telco-engineer: laptop-scale agentic "
                                "discovery of wireless algorithms")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="run a search from a YAML config")
    s.add_argument("config")
    s.add_argument("--run-dir", default=None, help="override run_dir from the config")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("demo", help="offline end-to-end demo with a scripted LLM (no API key)")
    s.add_argument("--run-dir", default="runs/demo")
    s.add_argument("--resume", action="store_true")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("eval", help="evaluate one candidate file")
    s.add_argument("file")
    s.add_argument("--task", default=str(DEFAULT_TASK))
    s.add_argument("--split", default="dev", choices=["dev", "test"])
    s.add_argument("--timeout", type=float, default=300.0)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("report", help="(re)build REPORT.md and report.png for a run")
    s.add_argument("run_dir")
    s.add_argument("--task", default=None)
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("compare", help="compare several runs on one plot")
    s.add_argument("run_dirs", nargs="+")
    s.add_argument("--out", default="runs/compare.png")
    s.add_argument("--task", default=None)
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("check-llm", help="send one tiny request to the configured models")
    s.add_argument("config")
    s.set_defaults(fn=cmd_check_llm)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
