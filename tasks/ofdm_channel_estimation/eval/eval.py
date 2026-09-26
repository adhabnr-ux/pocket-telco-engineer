"""
Evaluation harness for the statistics-agnostic OFDM channel estimation task.

A candidate is a Python file (default ``draft.py``) that defines

    estimate_channel(y, no) -> h_hat

    y     : torch.complex64 tensor [batch, NUM_OFDM_SYMBOLS, FFT_SIZE]
            Received resource grid (after CP removal and FFT) of a SISO link.
            Pilot REs carry ``PILOT_SYMBOLS`` (see link_config.py); all other
            REs carry unknown unit-energy 16-QAM data symbols.
    no    : torch.float32 tensor [batch]
            Noise variance per example (SNR per RE = 1 / no, channel has unit
            average energy).
    h_hat : torch.complex64 tensor (or numpy array) [batch, NUM_OFDM_SYMBOLS, FFT_SIZE]
            Estimate of the channel frequency response on *every* RE.

The channel is drawn from one of several 3GPP TR 38.901 TDL scenarios
(different delay spreads, Doppler, LOS/NLOS). The estimator is not told
which scenario produced a batch.

Metric: Normalized Validation Error (NVE), lower is better.

    NVE = mean over (scenario, SNR) of  MSE_candidate / MSE_LS-linear

where MSE_LS-linear is the reference least-squares estimator with linear
interpolation (``baselines/ls_linear.py``). NVE < 1 beats the baseline.
Complexity: median wall-clock time of one call on a batch of 32 grids
(seconds, lower is better).

Usage:
    python eval.py [source_file] [--split dev|test] [--json results.json]

First line of stdout is machine-parsed:
    SUCCESS, <nve>, <runtime_seconds>
    FAILURE,
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from link_config import (  # noqa: E402
    CARRIER_FREQUENCY,
    CYCLIC_PREFIX_LENGTH,
    DATA_NUM_BITS_PER_SYMBOL,
    FFT_SIZE,
    NUM_OFDM_SYMBOLS,
    PILOT_MASK,
    PILOT_SYMBOLS,
    SUBCARRIER_SPACING,
)

# ---------------------------------------------------------------------------
# Evaluation protocol (single source of truth)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    model: str            # TDL model: A, B, C (NLOS) or D, E (LOS)
    delay_spread: float   # [s]
    min_speed: float      # [m/s]
    max_speed: float      # [m/s]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("indoor_flat", "A", 30e-9, 0.0, 3.0),
    Scenario("urban_vehicular", "C", 300e-9, 0.0, 30.0),
    Scenario("high_speed", "B", 100e-9, 50.0, 100.0),
    Scenario("rural_los", "D", 30e-9, 0.0, 10.0),
)

SNRS_DB: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)

# dev: used during the search (the number agents see and optimize)
# test: held-out draws with different seeds, used only for final reporting
SPLITS = {
    "dev": {"seed": 1000, "batch_size": 32},
    "test": {"seed": 9000, "batch_size": 128},
}

TIMING_BATCH = 32
NUM_WARMUP_CALLS = 2
NUM_TIMING_RUNS = 10
BASELINE_FILE = "baseline_mse.json"


# ---------------------------------------------------------------------------
# Data generation (Sionna)
# ---------------------------------------------------------------------------


def _point_seed(split: str, s_idx: int, snr_idx: int) -> int:
    return SPLITS[split]["seed"] + 100 * s_idx + snr_idx


def generate_batch(scenario: Scenario, snr_db: float, batch_size: int, seed: int,
                   noiseless: bool = False):
    """Generate one batch ``(y, no, h)``, each [batch, 14, 72]. Deterministic in ``seed``."""
    import torch
    import sionna.phy
    from sionna.phy.channel import OFDMChannel
    from sionna.phy.channel.tr38901 import TDL
    from sionna.phy.mapping import QAMSource
    from sionna.phy.ofdm import ResourceGrid

    sionna.phy.config.seed = seed
    torch.manual_seed(seed)

    rg = ResourceGrid(
        num_ofdm_symbols=NUM_OFDM_SYMBOLS,
        fft_size=FFT_SIZE,
        subcarrier_spacing=SUBCARRIER_SPACING,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=CYCLIC_PREFIX_LENGTH,
    )
    tdl = TDL(
        model=scenario.model,
        delay_spread=scenario.delay_spread,
        carrier_frequency=CARRIER_FREQUENCY,
        min_speed=scenario.min_speed,
        max_speed=scenario.max_speed,
    )
    channel = OFDMChannel(tdl, rg, add_awgn=not noiseless, normalize_channel=True,
                          return_channel=True)

    mask = torch.as_tensor(PILOT_MASK)
    pilots = torch.as_tensor(PILOT_SYMBOLS)
    num_data = int((~mask).sum())
    data = QAMSource(DATA_NUM_BITS_PER_SYMBOL)([batch_size, num_data])

    x = pilots.expand(batch_size, -1, -1).clone()
    x[:, ~mask] = data.to(x.dtype)
    x = x.reshape(batch_size, 1, 1, NUM_OFDM_SYMBOLS, FFT_SIZE)

    no = torch.full([batch_size], 10.0 ** (-snr_db / 10.0), dtype=torch.float32)
    if noiseless:
        y, h = channel(x)
    else:
        y, h = channel(x, no.reshape(batch_size, 1, 1, 1, 1))
    y = y.reshape(batch_size, NUM_OFDM_SYMBOLS, FFT_SIZE).to(torch.complex64)
    h = h.reshape(batch_size, NUM_OFDM_SYMBOLS, FFT_SIZE).to(torch.complex64)
    return y, no, h


def iter_points(split: str):
    """Yield ``(scenario, snr_db, y, no, h)`` for every evaluation point of a split."""
    bs = SPLITS[split]["batch_size"]
    for s_idx, sc in enumerate(SCENARIOS):
        for snr_idx, snr in enumerate(SNRS_DB):
            y, no, h = generate_batch(sc, snr, bs, _point_seed(split, s_idx, snr_idx))
            yield sc, snr, y, no, h


def point_key(scenario: Scenario, snr_db: float) -> str:
    return f"{scenario.name}@{snr_db:g}dB"


# ---------------------------------------------------------------------------
# Candidate handling
# ---------------------------------------------------------------------------


class CandidateError(Exception):
    pass


# The candidate runs in the same process as the harness, so it could in
# principle reach the ground-truth channel through frame introspection, or
# read the reference results. These patterns are rejected outright. The
# held-out ``test`` split (different seeds) additionally guards against
# overfitting to the ``dev`` draws.
FORBIDDEN_PATTERNS = (
    "inspect", "_getframe", "f_back", "f_locals", "f_globals", "gc.", "import gc",
    "tb_frame", "settrace", "setprofile", "baseline_mse", "eval.py", "build_baseline",
    "import eval", "from eval", "sys.modules",
)

# Hard cap on per-call runtime (batch of 32 grids). Complexity is a secondary
# objective; this only stops pathological solutions.
MAX_RUNTIME_S = 0.5


def integrity_check(source: str) -> None:
    hits = [p for p in FORBIDDEN_PATTERNS if p in source]
    if hits:
        raise CandidateError(
            "candidate uses forbidden constructs " + ", ".join(repr(h) for h in hits)
            + " (introspection or access to evaluation internals is not allowed)"
        )


def load_candidate(source_file: str):
    path = Path(source_file).resolve()
    if not path.exists():
        raise CandidateError(f"{source_file} not found")
    integrity_check(path.read_text(errors="replace"))
    spec = importlib.util.spec_from_file_location(f"candidate_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise CandidateError(f"cannot import {source_file}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    fn = getattr(module, "estimate_channel", None)
    if not callable(fn):
        raise CandidateError(f"{source_file} must define a callable estimate_channel(y, no)")
    return fn


def _as_tensor(h_hat, batch_size: int):
    import torch

    if isinstance(h_hat, np.ndarray):
        h_hat = torch.from_numpy(h_hat)
    if not isinstance(h_hat, torch.Tensor):
        raise CandidateError(f"estimate_channel must return a tensor or ndarray, got {type(h_hat).__name__}")
    expected = (batch_size, NUM_OFDM_SYMBOLS, FFT_SIZE)
    if tuple(h_hat.shape) != expected:
        raise CandidateError(f"estimate_channel returned shape {tuple(h_hat.shape)}, expected {expected}")
    if not torch.is_complex(h_hat):
        raise CandidateError(f"estimate_channel must return a complex tensor, got dtype {h_hat.dtype}")
    h_hat = h_hat.detach().to(torch.complex64).cpu()
    if not torch.isfinite(torch.view_as_real(h_hat)).all():
        raise CandidateError("estimate_channel returned NaN or Inf values")
    return h_hat


def mse(h_hat, h) -> float:
    import torch

    return float(torch.mean(torch.abs(h_hat.to(torch.complex128) - h.to(torch.complex128)) ** 2))


def measure_mse(fn, split: str) -> dict[str, float]:
    """Return {point_key: MSE} for a callable ``fn(y, no)`` on a split."""
    import torch

    out = {}
    with torch.no_grad():
        for sc, snr, y, no, h in iter_points(split):
            h_hat = _as_tensor(fn(y.clone(), no.clone()), y.shape[0])
            out[point_key(sc, snr)] = mse(h_hat, h)
    return out


def measure_runtime(fn) -> float:
    import torch

    y, no, _ = generate_batch(SCENARIOS[1], 10.0, TIMING_BATCH, seed=424242)
    with torch.no_grad():
        for _ in range(NUM_WARMUP_CALLS):
            fn(y.clone(), no.clone())
        times = []
        for _ in range(NUM_TIMING_RUNS):
            t0 = time.perf_counter()
            fn(y.clone(), no.clone())
            times.append(time.perf_counter() - t0)
        return float(np.median(times))  # median: robust to scheduler noise


def load_baselines(path: Path | None = None) -> dict:
    path = path or (HERE / BASELINE_FILE)
    if not path.exists():
        path = Path.cwd() / BASELINE_FILE
    with open(path) as f:
        return json.load(f)


def score(candidate_mse: dict[str, float], baselines: dict, split: str) -> dict:
    """Compute NVE and auxiliary metrics from per-point MSEs."""
    ref = baselines[split]["ls_linear"]
    genie = baselines[split].get("lmmse_genie")
    ratios, gaps, rows = [], [], []
    for key, m in candidate_mse.items():
        r = m / ref[key]
        ratios.append(r)
        gap = None
        if genie is not None and ref[key] > genie[key]:
            gap = (ref[key] - m) / (ref[key] - genie[key])
            gaps.append(gap)
        rows.append({"point": key, "mse": m, "ratio_vs_ls_linear": r, "gap_closed_vs_lmmse": gap})
    return {
        "nve": float(np.mean(ratios)),
        "gain_db_vs_ls_linear": float(-10.0 * np.mean(np.log10(ratios))),
        "mean_gap_closed_vs_lmmse": float(np.mean(gaps)) if gaps else None,
        "points": rows,
    }


def _summary_table(result: dict) -> str:
    by_scen: dict[str, list[float]] = {}
    for row in result["points"]:
        by_scen.setdefault(row["point"].split("@")[0], []).append(row["ratio_vs_ls_linear"])
    lines = [f"  {name:<16} mean MSE ratio vs LS-linear: {np.mean(v):.4f}" for name, v in by_scen.items()]
    gc = result["mean_gap_closed_vs_lmmse"]
    lines.append(f"  gain vs LS-linear: {result['gain_db_vs_ls_linear']:+.2f} dB (geometric mean)")
    if gc is not None:
        lines.append(f"  fraction of the LS-linear -> genie-LMMSE gap closed: {gc:.3f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_file", nargs="?", default="draft.py")
    parser.add_argument("--split", choices=sorted(SPLITS), default="dev")
    parser.add_argument("--json", dest="json_path", default=None, help="write full results as JSON")
    parser.add_argument("--no-timing", action="store_true")
    args = parser.parse_args(argv)

    try:
        import torch

        threads = int(os.environ.get("PTE_EVAL_THREADS", "0")) or max(1, min(4, os.cpu_count() or 1))
        torch.set_num_threads(threads)
        fn = load_candidate(args.source_file)
        cand = measure_mse(fn, args.split)
        result = score(cand, load_baselines(), args.split)
        runtime = float("nan") if args.no_timing else measure_runtime(fn)
        if math.isfinite(runtime) and runtime > MAX_RUNTIME_S:
            raise CandidateError(
                f"too slow: {runtime * 1e3:.0f} ms per call on a batch of {TIMING_BATCH} "
                f"(limit {MAX_RUNTIME_S * 1e3:.0f} ms)"
            )
        result["runtime_s"] = runtime
        result["split"] = args.split
        if not math.isfinite(result["nve"]):
            raise CandidateError("non-finite NVE")
    except Exception as exc:  # report everything as a clean FAILURE
        print("FAILURE,")
        if isinstance(exc, CandidateError):
            print(f"Error: {exc}")
        else:
            exc_type, exc_value, exc_tb = sys.exc_info()
            print("".join(traceback.format_exception(exc_type, exc_value, exc_tb, limit=-4)).rstrip())
        return 1

    rt = result["runtime_s"]
    print(f"SUCCESS, {result['nve']:.6f}, {rt if math.isfinite(rt) else 0.0:.6e}")
    print(f"NVE ({args.split} split, lower is better, 1.0 = LS-linear baseline): {result['nve']:.4f}")
    print(_summary_table(result))
    if math.isfinite(rt):
        print(f"  runtime per call (batch {TIMING_BATCH}): {rt * 1e3:.2f} ms")
    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
