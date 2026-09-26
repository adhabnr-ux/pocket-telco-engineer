"""Append-only record of every evaluated candidate, plus summaries and Pareto fronts."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Candidate:
    id: str
    generation: int
    worker: int
    step: int
    idea: str
    success: bool
    metric: float | None
    complexity: float | None
    code_path: str
    spent_usd: float          # cumulative spend of the whole run when this candidate was evaluated
    llm_calls: int            # cumulative LLM calls of the whole run at that point
    wall_s: float             # seconds since the run started
    error: str = ""


class Leaderboard:
    def __init__(self, path: Path, higher_is_better: bool = False):
        self.path = Path(path)
        self.higher_is_better = higher_is_better
        self._lock = threading.Lock()
        self.items: list[Candidate] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.items.append(Candidate(**json.loads(line)))

    def add(self, cand: Candidate) -> None:
        with self._lock:
            self.items.append(cand)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(asdict(cand)) + "\n")

    def _key(self, c: Candidate) -> float:
        return -c.metric if self.higher_is_better else c.metric

    def successes(self) -> list[Candidate]:
        return [c for c in self.items if c.success and c.metric is not None]

    def best(self, k: int = 1) -> list[Candidate]:
        return sorted(self.successes(), key=self._key)[:k]

    def best_one(self) -> Candidate | None:
        b = self.best(1)
        return b[0] if b else None

    def pareto_front(self) -> list[Candidate]:
        """Candidates not dominated in (metric, complexity) — both 'lower is better' after sign fix."""
        pts = [c for c in self.successes() if c.complexity is not None]
        front = []
        for c in pts:
            dominated = any(
                (self._key(o) <= self._key(c) and o.complexity <= c.complexity)
                and (self._key(o) < self._key(c) or o.complexity < c.complexity)
                for o in pts
            )
            if not dominated:
                front.append(c)
        return sorted(front, key=lambda c: c.complexity)

    def best_so_far_curve(self) -> list[tuple[float, int, float, float]]:
        """[(spent_usd, llm_calls, wall_s, best_metric)] in evaluation order."""
        best = None
        curve = []
        for c in sorted(self.items, key=lambda c: (c.llm_calls, c.wall_s)):
            if c.success and c.metric is not None:
                improved = best is None or (c.metric > best if self.higher_is_better else c.metric < best)
                if improved:
                    best = c.metric
            if best is not None:
                curve.append((c.spent_usd, c.llm_calls, c.wall_s, best))
        return curve

    def summary(self, k: int = 8) -> str:
        """Compact text for the orchestrator: best ideas, and recent failures."""
        lines = []
        by_idea: dict[str, list[Candidate]] = {}
        for c in self.items:
            by_idea.setdefault(c.idea, []).append(c)
        scored = []
        for idea, cs in by_idea.items():
            ok = [c for c in cs if c.success and c.metric is not None]
            best = min(ok, key=self._key) if ok else None
            scored.append((idea, best, len(cs), len(ok), cs[-1].error))
        scored.sort(key=lambda t: (t[1] is None, self._key(t[1]) if t[1] else 0.0))
        for idea, best, n, n_ok, err in scored[:k]:
            if best:
                lines.append(f"- {idea}: best metric {best.metric:.4f}, runtime {best.complexity:.2e}s "
                             f"({n_ok}/{n} evaluations succeeded)")
            else:
                lines.append(f"- {idea}: all {n} evaluations FAILED (last error: {err[:160]})")
        return "\n".join(lines)
