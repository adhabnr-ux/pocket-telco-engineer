"""The task folder must also run unchanged inside NVIDIA's full AI Telco Engineer (AITE).

Set AITE_REPO=/path/to/the-ai-telco-engineer (and have langchain-core installed)
to run this test; it is skipped otherwise. A fake workspace executes the eval
command locally instead of in a Docker container.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import BASELINES, TASK

AITE = os.environ.get("AITE_REPO")
pytestmark = pytest.mark.skipif(not AITE or not Path(AITE, "src").is_dir(),
                                reason="set AITE_REPO to a checkout of NVlabs/the-ai-telco-engineer")


class _FakeContainer:
    def __init__(self, root: Path):
        self.root = root

    def exec(self, argv, timeout=None):
        p = subprocess.run([sys.executable, *argv[1:]], cwd=self.root, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode == 0, p.stdout + p.stderr


class _FakeWorkspace:
    def __init__(self, root: Path):
        self._host_workspace_path = root
        self._container = _FakeContainer(root)

    def _write_file(self, path, content):
        (self._host_workspace_path / path).write_text(content)

    def _write_file_binary(self, path, content):
        (self._host_workspace_path / path).write_bytes(content)

    def _delete(self, path):
        (self._host_workspace_path / path).unlink(missing_ok=True)


def test_eval_tool_runs_in_aite(tmp_path):
    pytest.importorskip("langchain_core")
    sys.path.insert(0, str(Path(AITE, "src")))
    import importlib.util

    spec = importlib.util.spec_from_file_location("aite_eval_tool", TASK / "eval_tool.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = json.loads((TASK / "config.json").read_text())
    tool = mod.EvalTool(eval_timeout=cfg["tools_config"]["eval_timeout"], higher_is_better=cfg["higher_is_better"])
    ws = _FakeWorkspace(tmp_path)
    tool.set_workspace(ws)
    assert (tmp_path / "link_config.py").exists()           # visible to the agent from the start
    (tmp_path / "draft.py").write_text((BASELINES / "dft_denoise.py").read_text())

    [t] = tool.get_tools()
    assert t.name == "evaluate_estimator"
    out = t.invoke({})
    first = out.strip().splitlines()[0]
    m = tool.RESULT_RE.match(first)
    assert m and m.group(1) == "SUCCESS", out
    assert float(m.group(2)) == pytest.approx(1.2126, abs=5e-3)
    assert (tmp_path / "solution.py").exists()             # AITE auto-saves a new best
    assert not (tmp_path / "eval.py").exists()             # harness cleaned up after evaluation
    assert not (tmp_path / "baseline_mse.json").exists()
