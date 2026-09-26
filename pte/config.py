"""Run configuration (YAML) with ${ENV_VAR:-default} interpolation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from .llm import ModelSpec

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate(value):
    if isinstance(value, str):
        if not _ENV_RE.search(value):
            return value

        def repl(m):
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")
        out = _ENV_RE.sub(repl, value)
        # keep numbers numeric after substitution
        try:
            return int(out) if out.strip().lstrip("-").isdigit() else float(out) if _is_float(out) else out
        except ValueError:
            return out
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _is_float(s: str) -> bool:
    try:
        float(s)
        return any(ch in s for ch in ".eE") and s.strip() != ""
    except ValueError:
        return False


@dataclass
class SearchConfig:
    generations: int = 3          # orchestrator rounds
    ideas_per_generation: int = 3
    workers_per_generation: int = 3
    steps_per_worker: int = 4     # LLM write -> evaluate iterations per worker
    max_parallel: int = 2         # concurrent workers (evaluations are CPU-bound)
    seed_with_best: bool = True   # show workers the current best solution
    keep_last_drafts: int = 2     # older drafts in a worker's context are elided to save tokens


@dataclass
class BudgetConfig:
    max_usd: float = 2.0
    max_calls: int = 500


@dataclass
class EvalConfig:
    timeout_s: float = 180.0
    memory_mb: int = 16384
    final_top_k: int = 5          # re-evaluate the best k on the held-out split at the end


@dataclass
class RunConfig:
    task: str
    run_dir: str
    orchestrator: ModelSpec
    worker: ModelSpec
    search: SearchConfig = field(default_factory=SearchConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 0
    notes: str = ""


def _build(cls, data: dict | None):
    data = data or {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = _interpolate(yaml.safe_load(path.read_text()))
    base = path.parent

    def resolve(p: str) -> str:
        q = Path(p)
        return str(q if q.is_absolute() else (base / q).resolve())

    models = raw.get("models", {})
    worker = _build(ModelSpec, models.get("worker"))
    orchestrator = _build(ModelSpec, models.get("orchestrator") or models.get("worker"))
    return RunConfig(
        task=resolve(raw["task"]),
        run_dir=resolve(raw.get("run_dir", "runs/" + path.stem)),
        orchestrator=orchestrator,
        worker=worker,
        search=_build(SearchConfig, raw.get("search")),
        budget=_build(BudgetConfig, raw.get("budget")),
        eval=_build(EvalConfig, raw.get("eval")),
        seed=int(raw.get("seed", 0)),
        notes=str(raw.get("notes", "")),
    )
