"""Run untrusted candidate code in a subprocess with a timeout and a scrubbed environment.

This is *not* a security boundary like the Docker containers used by the full
AI Telco Engineer: it prevents accidents (runaway loops, memory blow-ups,
leaking your API keys into candidate code), not a determined attacker. Only
run models you trust on a machine you do not mind them touching, or use the
Docker path of the full framework.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
MAX_OUTPUT_CHARS = 6000


@dataclass
class ProcResult:
    returncode: int | None
    output: str
    timed_out: bool


def scrubbed_env() -> dict[str, str]:
    """Copy of the environment without anything that looks like a credential."""
    env = {k: v for k, v in os.environ.items() if not any(m in k.upper() for m in _SECRET_MARKERS)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "4")
    return env


def _limit_resources(memory_mb: int | None):
    def _apply():  # runs in the child between fork and exec
        os.setsid()
        if memory_mb:
            try:
                import resource

                limit = memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except (ImportError, ValueError, OSError):
                pass  # not supported (e.g. macOS ignores RLIMIT_AS); timeout still applies
    return _apply


def run_python(args: list[str], cwd: Path, timeout_s: float, memory_mb: int | None = 16384,
               extra_env: dict[str, str] | None = None) -> ProcResult:
    """Run ``python <args>`` in ``cwd``; kill the whole process group on timeout."""
    env = scrubbed_env()
    env.update(extra_env or {})
    preexec = _limit_resources(memory_mb) if os.name == "posix" else None
    proc = subprocess.Popen(
        [sys.executable, *args],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=preexec,
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        return ProcResult(proc.returncode, _truncate(out), False)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        out, _ = proc.communicate()
        return ProcResult(None, _truncate(out or ""), True)


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return text[:half] + "\n... [output truncated] ...\n" + text[-half:]
