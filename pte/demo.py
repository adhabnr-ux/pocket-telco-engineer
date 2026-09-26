"""Offline demo: a scripted 'LLM' walks through the whole pipeline without any API key.

It is not intelligent. It replays a fixed sequence of drafts (one broken, the
references, and a simple SNR-switching hybrid) so you can see logging,
failure handling, the leaderboard, held-out re-evaluation and the report
before spending a cent on a real model.
"""

from __future__ import annotations

import json
from pathlib import Path

from .llm import Message, Scripted

BROKEN = '''\
import torch

def estimate_channel(y, no):
    # Bug on purpose: returns only the pilot symbols -> wrong shape
    return y[:, [2, 11], :]
'''


def _hybrid(task_folder: Path) -> str:
    """SNR-switching hybrid: DFT denoising at low SNR (it removes noise well), linear
    interpolation at high SNR (DFT has an error floor from band-edge leakage).
    Built by inlining the two reference files so the candidate is self-contained."""
    lin = (task_folder / "baselines" / "ls_linear.py").read_text()
    dft = (task_folder / "baselines" / "dft_denoise.py").read_text()
    return (
        "# --- inlined reference: ls_linear ---\n"
        + lin.replace("def estimate_channel", "def _lin")
        + "\n# --- inlined reference: dft_denoise ---\n"
        + dft.replace("def estimate_channel", "def _dft").replace("def _denoise_and_interpolate", "def _dft_core")
             .replace("_denoise_and_interpolate(", "_dft_core(")
        + "\n# --- hybrid ---\nSNR_SWITCH_DB = 12.0\n\n"
        + "def estimate_channel(y, no):\n"
        + "    snr_db = -10.0 * torch.log10(no.mean())\n"
        + "    return _dft(y, no) if snr_db < SNR_SWITCH_DB else _lin(y, no)\n"
    )


def scripted_llms(task_folder: str | Path) -> tuple[Scripted, Scripted]:
    task_folder = Path(task_folder)
    drafts = [
        BROKEN,
        (task_folder / "baselines" / "ls_linear.py").read_text(),
        (task_folder / "baselines" / "dft_denoise.py").read_text(),
        _hybrid(task_folder),
    ]

    def orchestrator(system: str, messages: list[Message]) -> str:
        return json.dumps([
            {"title": "Linear interpolation baseline",
             "description": "LS at pilots, linear interpolation in frequency and time."},
            {"title": "SNR-switched delay-domain denoising",
             "description": "Use DFT denoising at low SNR, linear interpolation at high SNR."},
        ])

    def worker(system: str, messages: list[Message]) -> str:
        step = sum(1 for m in messages if m.role == "assistant")
        code = drafts[min(step, len(drafts) - 1)]
        return f"Draft {step + 1}.\n```python\n{code}```"

    # Pretend prices so the spend axis of the report is exercised (nothing is charged).
    return (Scripted(orchestrator, name="scripted-orchestrator", price_in_per_mtok=1.0, price_out_per_mtok=5.0),
            Scripted(worker, name="scripted-worker", price_in_per_mtok=1.0, price_out_per_mtok=5.0))
