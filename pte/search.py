"""The search loop: orchestrator proposes ideas, workers implement and refine them.

A laptop-scale re-implementation of the idea-driven loop of NVIDIA's
"The AI Telco Engineer" (AITE): no Docker, no GPU, no tool-calling, any
OpenAI-compatible or Anthropic model, a hard dollar cap, and full logging so
that runs can be compared as a function of spend.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from . import prompts
from .budget import Budget, BudgetExceeded
from .config import RunConfig
from .leaderboard import Candidate, Leaderboard
from .llm import LLM, LLMError, Message
from .task import EvalResult, Task

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?[ \t]*\n.*?```", re.DOTALL | re.IGNORECASE)


def _elide_old_drafts(messages: list[Message], keep_last: int) -> list[Message]:
    """Replace the code of older assistant drafts with a placeholder to bound context size."""
    idx = [i for i, m in enumerate(messages) if m.role == "assistant"]
    old = set(idx[:-keep_last]) if keep_last > 0 else set(idx)
    out = []
    for i, m in enumerate(messages):
        if i in old:
            out.append(Message(m.role, _CODE_BLOCK_RE.sub("```python\n# [earlier draft elided]\n```", m.content)))
        else:
            out.append(m)
    return out


class Search:
    def __init__(self, cfg: RunConfig, task: Task, orchestrator: LLM, worker: LLM,
                 log: Callable[[str], None] = print):
        self.cfg = cfg
        self.task = task
        self.orchestrator = orchestrator
        self.worker_llm = worker
        self.log = log
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "candidates").mkdir(exist_ok=True)
        (self.run_dir / "transcripts").mkdir(exist_ok=True)
        self.board = Leaderboard(self.run_dir / "leaderboard.jsonl", task.higher_is_better)
        self.budget = Budget(cfg.budget.max_usd, cfg.budget.max_calls)
        self.state_path = self.run_dir / "state.json"
        self.state = {"completed_generations": 0, "ideas": [], "elapsed_s": 0.0, "stopped_reason": None}
        if self.state_path.exists():
            self.state.update(json.loads(self.state_path.read_text()))
            self.budget.restore(self.state.get("budget", {}))
        self._stop = threading.Event()
        # Split CPU cores between concurrent evaluations so runtimes are not inflated by oversubscription.
        self._eval_threads = max(1, (os.cpu_count() or 1) // max(1, cfg.search.max_parallel))
        self._t0 = time.time() - float(self.state.get("elapsed_s", 0.0))

    # ------------------------------------------------------------------ utils
    def _elapsed(self) -> float:
        return time.time() - self._t0

    def _save_state(self) -> None:
        self.state["budget"] = self.budget.snapshot()
        self.state["elapsed_s"] = self._elapsed()
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def _call(self, llm: LLM, system: str, messages: list[Message]):
        self.budget.check()
        reply = llm.complete(system, messages)
        cost = self.budget.record(reply.input_tokens, reply.output_tokens,
                                  llm.price_in_per_mtok, llm.price_out_per_mtok)
        return reply, cost

    # ------------------------------------------------------------ orchestrator
    def propose_ideas(self, generation: int) -> list[dict]:
        n = self.cfg.search.ideas_per_generation
        req = prompts.orchestrator_request(self.task.prompt, n, self.board.summary(), generation)
        try:
            reply, _ = self._call(self.orchestrator, prompts.ORCHESTRATOR_SYSTEM, [Message("user", req)])
            ideas = prompts.parse_ideas(reply.text, n)
        except LLMError as exc:
            self.log(f"[orchestrator] LLM error, using fallback ideas: {exc}")
            ideas = []
        if not ideas:
            self.log("[orchestrator] could not parse ideas, using fallback ideas")
            start = (generation * n) % len(prompts.FALLBACK_IDEAS)
            ideas = [prompts.FALLBACK_IDEAS[(start + i) % len(prompts.FALLBACK_IDEAS)] for i in range(n)]
        return ideas

    # ----------------------------------------------------------------- worker
    def _run_worker(self, generation: int, worker: int, idea: dict) -> None:
        s = self.cfg.search
        tag = f"g{generation:02d}_w{worker:02d}"
        # A generation interrupted by Ctrl+C / budget is re-run on resume: never overwrite old files.
        n = 1
        while (self.run_dir / "transcripts" / f"{tag}.json").exists() or any(
                (self.run_dir / "candidates").glob(f"{tag}_s*.py")):
            n += 1
            tag = f"g{generation:02d}_w{worker:02d}_r{n}"
        best = self.board.best_one() if s.seed_with_best else None
        best_code = Path(best.code_path).read_text() if best and Path(best.code_path).exists() else None
        messages = [Message("user", prompts.worker_request(
            self.task.prompt, idea, best_code, best.metric if best else None))]
        local_best: float | None = None
        transcript = []
        for step in range(1, s.steps_per_worker + 1):
            if self._stop.is_set():
                break
            try:
                reply, cost = self._call(self.worker_llm, prompts.WORKER_SYSTEM,
                                         _elide_old_drafts(messages, s.keep_last_drafts))
            except BudgetExceeded as exc:
                self._stop.set()
                self.state["stopped_reason"] = str(exc)
                self.log(f"[{tag}] stopping: {exc}")
                break
            except LLMError as exc:
                self.log(f"[{tag}] LLM error: {exc}")
                break
            messages.append(Message("assistant", reply.text))
            transcript.append({"step": step, "reply": reply.text, "input_tokens": reply.input_tokens,
                               "output_tokens": reply.output_tokens, "cost_usd": cost})

            code = prompts.extract_code(reply.text)
            code_path = self.run_dir / "candidates" / f"{tag}_s{step:02d}.py"
            if code is None:
                result = EvalResult(False, None, None,
                                    "FAILURE,\nNo ```python code block found in your reply. "
                                    "Reply with the complete draft.py in one ```python block.")
            else:
                code_path.write_text(code)
                result = self.task.evaluate_code(code, split="dev", timeout_s=self.cfg.eval.timeout_s,
                                                 memory_mb=self.cfg.eval.memory_mb, threads=self._eval_threads)
            snap = self.budget.snapshot()
            err = "" if result.success else _first_error_line(result.output)
            self.board.add(Candidate(
                id=f"{tag}_s{step:02d}", generation=generation, worker=worker, step=step,
                idea=idea.get("title", "idea"), success=result.success, metric=result.metric,
                complexity=result.complexity, code_path=str(code_path) if code else "",
                spent_usd=snap["spent_usd"], llm_calls=snap["calls"], wall_s=round(self._elapsed(), 2),
                error=err,
            ))
            if result.success and self.task.better(result.metric, local_best):
                local_best = result.metric
            status = f"metric={result.metric:.4f}" if result.success else f"FAILED ({err[:80]})"
            self.log(f"[{tag}] step {step}/{s.steps_per_worker} {status}  spent=${snap['spent_usd']:.4f}")
            messages.append(Message("user", prompts.worker_feedback(
                result.feedback(), step, s.steps_per_worker, local_best)))
            transcript[-1]["feedback"] = result.feedback()
        (self.run_dir / "transcripts" / f"{tag}.json").write_text(
            json.dumps({"idea": idea, "turns": transcript}, indent=1))

    # -------------------------------------------------------------------- run
    def run(self) -> dict:
        s = self.cfg.search
        start_gen = int(self.state["completed_generations"])
        if start_gen:
            self.log(f"resuming after generation {start_gen} (spent ${self.budget.spent_usd:.4f})")
        for gen in range(start_gen, s.generations):
            if self._stop.is_set():
                break
            try:
                ideas = self.propose_ideas(gen)
            except BudgetExceeded as exc:
                self.state["stopped_reason"] = str(exc)
                self.log(f"stopping: {exc}")
                break
            self.state["ideas"].append({"generation": gen, "ideas": ideas})
            self.log(f"=== generation {gen + 1}/{s.generations}: " + " | ".join(i["title"] for i in ideas))
            jobs = [(gen, w, ideas[w % len(ideas)]) for w in range(s.workers_per_generation)]
            with ThreadPoolExecutor(max_workers=max(1, s.max_parallel)) as pool:
                list(pool.map(lambda j: self._run_worker(*j), jobs))
            if not self._stop.is_set():
                self.state["completed_generations"] = gen + 1
            self._save_state()
            best = self.board.best_one()
            if best:
                self.log(f"--- after generation {gen + 1}: best {best.metric:.4f} ({best.idea})")
        self._save_state()
        return self.finalize()

    def finalize(self) -> dict:
        """Re-evaluate the best distinct candidates on the held-out test split."""
        seen, finals = set(), []
        for c in self.board.best(len(self.board.items)):
            if len(finals) >= self.cfg.eval.final_top_k:
                break
            code = Path(c.code_path).read_text()
            h = hashlib.sha256(code.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            test = self.task.evaluate_file(Path(c.code_path), split="test", timeout_s=self.cfg.eval.timeout_s * 3,
                                           memory_mb=self.cfg.eval.memory_mb)
            finals.append({"candidate": asdict(c), "dev_metric": c.metric,
                           "test_metric": test.metric, "test_success": test.success,
                           "test_details": {k: v for k, v in test.details.items() if k != "points"}})
            self.log(f"[final] {c.id}: dev {c.metric:.4f} -> held-out test "
                     + (f"{test.metric:.4f}" if test.success else "FAILED"))
        summary = {
            "task": self.task.name,
            "orchestrator_model": self.orchestrator.name,
            "worker_model": self.worker_llm.name,
            "budget": self.budget.snapshot(),
            "max_usd": self.cfg.budget.max_usd,
            "stopped_reason": self.state.get("stopped_reason"),
            "num_candidates": len(self.board.items),
            "num_successes": len(self.board.successes()),
            "elapsed_s": round(self._elapsed(), 1),
            "final": finals,
            "seed": self.cfg.seed,
            "notes": self.cfg.notes,
        }
        (self.run_dir / "final.json").write_text(json.dumps(summary, indent=2))
        return summary


def _first_error_line(output: str) -> str:
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.upper().startswith("FAILURE")]
    if not lines:
        return "unknown error"
    for ln in reversed(lines):  # last exception line is usually the informative one
        if "Error" in ln or "error" in ln or "Exception" in ln:
            return ln[:300]
    return lines[-1][:300]
