"""
AITE (NVIDIA "The AI Telco Engineer") evaluation tool for the
statistics-agnostic OFDM channel estimation task.

Drop this whole task folder into ``the-ai-telco-engineer/tasks/`` to run the
task with the full multi-worker framework on a GPU box; pocket-telco-engineer
runs the very same ``eval/eval.py`` on a laptop.

Output format (first line only is parsed by the framework):
``SUCCESS, <metric>, <complexity>`` or ``FAILURE,``.
"""

from pathlib import Path

from langchain_core.tools import BaseTool, tool
from tool_lib.base import EvalToolBase

_EVAL_DIR = Path(__file__).parent / "eval"
_STAGED = ("eval.py", "link_config.py", "baseline_mse.json")


class EvalTool(EvalToolBase):
    _TOOL_DESCRIPTION = """\
Evaluate the OFDM channel estimator.

Expects `draft.py` in the workspace defining `estimate_channel(y, no)` with
y: complex64 [batch, 14, 72], no: float32 [batch], returning a complex
[batch, 14, 72] channel estimate. See the task prompt for rules.

Returns:
    str: First line ``SUCCESS, <metric>, <complexity>`` or ``FAILURE,``.
    <metric> is the Normalized Validation Error (lower is better, 1.0 = LS-linear reference).
    <complexity> is the average per-call runtime in seconds (lower is better).
    Remaining lines are per-scenario diagnostics."""

    def __init__(self, eval_timeout: int, **kwargs):
        super().__init__(eval_timeout, **kwargs)

    def set_workspace(self, workspace) -> None:
        super().set_workspace(workspace)
        # link_config.py is part of the task definition: keep it visible at all times.
        workspace._write_file("link_config.py", (_EVAL_DIR / "link_config.py").read_text())

    def setup_workspace(self) -> None:
        if self._workspace is None:
            raise ValueError("Workspace not set")
        for name in _STAGED:
            self._workspace._write_file(name, (_EVAL_DIR / name).read_text())

    def cleanup_workspace(self) -> None:
        self._workspace._delete("eval.py")
        self._workspace._delete("baseline_mse.json")

    # _execute is inherited: it runs ``python eval.py <filename>`` inside the worker container.

    def _create_tools(self) -> list[BaseTool]:
        def _evaluate() -> str:
            """Evaluate the OFDM channel estimator."""
            return self.run_evaluation(self.default_source_file)

        evaluate_estimator = tool(_evaluate)
        evaluate_estimator.name = "evaluate_estimator"
        evaluate_estimator.description = self._TOOL_DESCRIPTION
        return [evaluate_estimator]
