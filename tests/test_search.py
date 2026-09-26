"""End-to-end search with scripted LLMs: logging, budget stop, resume, held-out finals, report."""

import json
from pathlib import Path

import pytest

from conftest import BASELINES, TASK
from pte.config import BudgetConfig, EvalConfig, RunConfig, SearchConfig
from pte.llm import ModelSpec, Scripted
from pte.search import Search

GOOD = (BASELINES / "ls_linear.py").read_text()
BETTER = (BASELINES / "dft_denoise.py").read_text().replace(
    "_NUM_CAUSAL_TAPS = 10", "_NUM_CAUSAL_TAPS = 10  # tweak")


def _orch(system, messages):
    return '[{"title": "Idea A", "description": "do A"}, {"title": "Idea B", "description": "do B"}]'


def _cfg(tmp_path, **search_kw):
    kw = dict(generations=1, ideas_per_generation=2, workers_per_generation=1, steps_per_worker=3,
              max_parallel=1)
    kw.update(search_kw)
    return RunConfig(task=str(TASK), run_dir=str(tmp_path / "run"),
                     orchestrator=ModelSpec(provider="scripted", model="o"),
                     worker=ModelSpec(provider="scripted", model="w"),
                     search=SearchConfig(**kw), budget=BudgetConfig(max_usd=10.0, max_calls=100),
                     eval=EvalConfig(timeout_s=120, final_top_k=2))


def test_search_end_to_end(tmp_path, task):
    replies = iter(["I will think first, no code yet.",
                    f"```python\n{GOOD}```",
                    f"Better:\n```python\n{BETTER}```"])
    seen_feedback = []

    def worker(system, messages):
        if len(messages) > 1:
            seen_feedback.append(messages[-1].content)
        return next(replies)

    s = Search(_cfg(tmp_path), task, Scripted(_orch, "o", 1.0, 1.0), Scripted(worker, "w", 1.0, 1.0),
               log=lambda m: None)
    summary = s.run()
    items = s.board.items
    assert [c.success for c in items] == [False, True, True]
    assert "No ```python code block" in seen_feedback[0]
    assert "SUCCESS, 1.000000" in seen_feedback[1]
    assert items[1].metric == pytest.approx(1.0)
    assert summary["num_candidates"] == 3 and summary["num_successes"] == 2
    assert summary["budget"]["calls"] == 4 and summary["budget"]["spent_usd"] > 0
    finals = summary["final"]
    assert len(finals) == 2 and all(f["test_success"] for f in finals)
    run = Path(s.cfg.run_dir)
    assert json.loads((run / "state.json").read_text())["completed_generations"] == 1
    assert (run / "transcripts" / "g00_w00.json").exists()

    from pte.report import compare_runs, run_report

    md = run_report(run, task)
    assert md.exists() and (run / "report.png").exists()
    text = md.read_text()
    assert "lmmse_genie" in text and "Idea A" in text
    out = compare_runs([run], task, tmp_path / "cmp.png")
    assert out.exists() and out.with_suffix(".md").exists()


def test_budget_cap_stops_search_and_resume_continues(tmp_path, task):
    cfg = _cfg(tmp_path, generations=2, steps_per_worker=2)
    cfg.budget = BudgetConfig(max_usd=10.0, max_calls=3)       # orchestrator + 2 worker calls
    worker = Scripted(lambda s, m: f"```python\n{GOOD}```", "w")
    s = Search(cfg, task, Scripted(_orch, "o"), worker, log=lambda m: None)
    summary = s.run()
    assert "call cap" in (summary["stopped_reason"] or "")
    assert summary["budget"]["calls"] == 3
    state = json.loads((Path(cfg.run_dir) / "state.json").read_text())
    assert state["completed_generations"] == 1                 # gen 1 finished, gen 2 never started

    cfg.budget = BudgetConfig(max_usd=10.0, max_calls=100)     # raise the cap and resume
    s2 = Search(cfg, task, Scripted(_orch, "o"), worker, log=lambda m: None)
    assert s2.budget.calls == 3                                # spend restored from state
    s2.run()
    state = json.loads((Path(cfg.run_dir) / "state.json").read_text())
    assert state["completed_generations"] == 2
    assert len(s2.board.items) == 4


def test_orchestrator_garbage_falls_back(tmp_path, task):
    s = Search(_cfg(tmp_path), task, Scripted(lambda s, m: "no json", "o"),
               Scripted(lambda s, m: "nothing", "w"), log=lambda m: None)
    ideas = s.propose_ideas(0)
    assert len(ideas) == 2 and all(i["title"] and i["description"] for i in ideas)
