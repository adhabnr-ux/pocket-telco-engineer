"""Load an AITE-format task folder and evaluate candidates against it."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .sandbox import run_python

RESULT_RE = re.compile(
    r"^(SUCCESS|FAILURE)\s*,"
    r"(?:\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*,"
    r"\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?))?\s*$",
    re.IGNORECASE,
)


@dataclass
class EvalResult:
    success: bool
    metric: float | None
    complexity: float | None
    output: str
    details: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, output: str, details: dict | None = None) -> "EvalResult":
        lines = output.strip().splitlines()
        # The harness prints the status line first, but libraries may emit warnings
        # before it; take the first line that matches the protocol.
        for line in lines:
            m = RESULT_RE.match(line.strip())
            if m:
                ok = m.group(1).upper() == "SUCCESS" and m.group(2) is not None
                return cls(
                    success=ok,
                    metric=float(m.group(2)) if ok else None,
                    complexity=float(m.group(3)) if ok else None,
                    output=output,
                    details=details or {},
                )
        return cls(False, None, None, output or "(no output)", details or {})

    def feedback(self, max_chars: int = 2500) -> str:
        text = self.output.strip()
        return text if len(text) <= max_chars else text[:max_chars] + "\n... [truncated]"


class Task:
    """An AITE-compatible task folder (prompt.md, config.json, eval/eval.py, ...)."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder).resolve()
        if not (self.folder / "eval" / "eval.py").exists():
            raise FileNotFoundError(f"{self.folder} is not a task folder (missing eval/eval.py)")
        cfg_path = self.folder / "config.json"
        self.config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.higher_is_better: bool = bool(self.config.get("higher_is_better", False))
        self.prompt: str = (self.folder / self.config.get("prompt_path", "prompt.md")).read_text()
        self.name = self.folder.name

    # Files the candidate may read (task definition) vs harness-only files.
    @property
    def public_files(self) -> list[Path]:
        return [p for p in (self.folder / "eval").glob("link_config.py")]

    @property
    def harness_files(self) -> list[Path]:
        skip = {"build_baseline.py", "__pycache__"}
        return [p for p in (self.folder / "eval").iterdir() if p.name not in skip and p.is_file()]

    def better(self, a: float, b: float | None) -> bool:
        if b is None:
            return True
        return a > b if self.higher_is_better else a < b

    def prepare_workspace(self, workspace: Path) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        for p in self.public_files:
            shutil.copy2(p, workspace / p.name)

    def evaluate_file(self, source: Path, split: str = "dev", timeout_s: float = 180.0,
                      memory_mb: int | None = 16384, threads: int | None = None) -> EvalResult:
        """Evaluate a candidate file in a fresh scratch directory (harness copied in)."""
        source = Path(source).resolve()
        with tempfile.TemporaryDirectory(prefix="pte_eval_") as tmp:
            tmp = Path(tmp)
            for p in self.harness_files:
                shutil.copy2(p, tmp / p.name)
            shutil.copy2(source, tmp / "draft.py")
            json_path = tmp / "_result.json"
            proc = run_python(["eval.py", "draft.py", "--split", split, "--json", json_path.name],
                              cwd=tmp, timeout_s=timeout_s, memory_mb=memory_mb,
                              extra_env={"PTE_EVAL_THREADS": str(threads)} if threads else None)
            if proc.timed_out:
                return EvalResult(False, None, None,
                                  f"FAILURE,\nEvaluation timed out after {timeout_s:.0f}s.\n{proc.output[-1500:]}")
            details = {}
            if json_path.exists():
                try:
                    details = json.loads(json_path.read_text())
                except json.JSONDecodeError:
                    details = {}
            return EvalResult.parse(proc.output, details)

    def evaluate_code(self, code: str, split: str = "dev", timeout_s: float = 180.0,
                      memory_mb: int | None = 16384, threads: int | None = None) -> EvalResult:
        with tempfile.TemporaryDirectory(prefix="pte_src_") as tmp:
            src = Path(tmp) / "draft.py"
            src.write_text(code)
            return self.evaluate_file(src, split=split, timeout_s=timeout_s, memory_mb=memory_mb,
                                      threads=threads)

    def baselines(self) -> dict:
        path = self.folder / "eval" / "baseline_mse.json"
        return json.loads(path.read_text()) if path.exists() else {}
