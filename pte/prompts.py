"""Prompts and response parsing for the orchestrator (idea generator) and workers (implementers)."""

from __future__ import annotations

import json
import re

ORCHESTRATOR_SYSTEM = """\
You are the research lead of a team of wireless communications engineers.
You propose distinct, concrete algorithmic approaches for a signal processing
task. Each idea must be implementable in a single Python file within a few
iterations. Favour diversity: classical estimation theory, adaptive and
data-driven statistics, robust fallbacks, and hybrids. Learn from the
leaderboard: extend what works, fix what fails, and avoid repeating ideas
that already failed without a new twist."""

WORKER_SYSTEM = """\
You are an expert wireless communications and signal processing engineer who
writes correct, efficient PyTorch code. You implement the assigned approach
for the task below and improve it using the evaluation feedback you receive.

Every reply MUST contain exactly one complete Python file in a single
```python fenced code block. That file fully replaces draft.py. Keep any
reasoning short and put it before the code block."""


def orchestrator_request(task_prompt: str, n_ideas: int, leaderboard_summary: str, generation: int) -> str:
    history = leaderboard_summary or "No candidates have been evaluated yet."
    return f"""# Task
{task_prompt}

# Leaderboard so far (generation {generation})
{history}

# Your job
Propose {n_ideas} distinct algorithmic ideas for the next generation.
Reply with ONLY a JSON array of {n_ideas} objects, each with the keys
"title" (max 10 words) and "description" (3-6 sentences with enough technical
detail for an engineer to implement it). No prose outside the JSON."""


def worker_request(task_prompt: str, idea: dict, best_code: str | None, best_metric: float | None) -> str:
    seed = ""
    if best_code:
        seed = (f"\n\n# Current best solution in the whole search (metric {best_metric:.4f})\n"
                f"You may reuse parts of it.\n```python\n{best_code}\n```")
    return f"""# Task
{task_prompt}

# Assigned approach: {idea.get('title', 'untitled')}
{idea.get('description', '')}{seed}

Write the complete draft.py implementing this approach."""


def worker_feedback(result_text: str, step: int, steps_total: int, best_metric: float | None) -> str:
    best = f"Your best metric so far: {best_metric:.4f}." if best_metric is not None else "No successful evaluation yet."
    return f"""Evaluation result of your draft.py:
```
{result_text}
```
{best} Iteration {step}/{steps_total}. Fix any error, then improve the metric.
Reply with the complete updated draft.py in one ```python block."""


_CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str | None:
    """Return the last fenced code block that looks like the candidate file."""
    blocks = [b.strip() for b in _CODE_RE.findall(text or "")]
    blocks = [b for b in blocks if b]
    if not blocks:
        # tolerate an unterminated final fence (common when max_tokens is hit)
        m = re.search(r"```(?:python|py)?[ \t]*\n(.*)$", text or "", re.DOTALL | re.IGNORECASE)
        if m and m.group(1).strip():
            blocks = [m.group(1).strip()]
    if not blocks:
        return None
    with_fn = [b for b in blocks if "def " in b]
    return (with_fn or blocks)[-1] + "\n"


def parse_ideas(text: str, n: int) -> list[dict]:
    """Parse the orchestrator's JSON (tolerating code fences and chatter)."""
    candidates = []
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text or "", re.DOTALL)
    candidates.extend(fenced)
    m = re.search(r"\[.*\]", text or "", re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            data = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = data.get("ideas", [data])
        ideas = [
            {"title": str(d.get("title", "idea")).strip()[:120], "description": str(d.get("description", "")).strip()}
            for d in data if isinstance(d, dict) and (d.get("title") or d.get("description"))
        ]
        if ideas:
            return ideas[:n]
    return []


FALLBACK_IDEAS = [
    {"title": "Delay-domain denoising with adaptive window",
     "description": "Estimate LS at the pilots, transform to the delay domain, keep taps above a noise-adaptive "
                    "threshold inside an adaptive window, and transform back. Interpolate linearly in time."},
    {"title": "Data-driven frequency-domain LMMSE",
     "description": "Estimate the frequency correlation of the channel from the LS pilot estimates across the batch "
                    "and use it in an LMMSE interpolator with the known noise variance, then interpolate in time."},
    {"title": "Robust hybrid with per-batch model selection",
     "description": "Compute several candidate estimates (linear, denoised, smoothed) and pick or blend them per "
                    "batch using a residual-based estimate of their error on the pilots (e.g. leave-one-out)."},
    {"title": "Doppler-aware time interpolation",
     "description": "Estimate the temporal correlation between the two pilot symbols from the data and use it to "
                    "choose between linear interpolation, averaging and extrapolation across OFDM symbols."},
]
