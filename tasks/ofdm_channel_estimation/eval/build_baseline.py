"""
Build ``baseline_mse.json``: per-point MSE of the reference estimators on
the dev and test splits.

Estimators:
  ls_nearest   - LS + nearest-neighbour interpolation (weak)
  ls_linear    - LS + linear interpolation (the metric normalizer, NVE = 1)
  dft_denoise  - LS + delay-domain denoising (hand-written "engineer" baseline)
  lmmse_genie  - 2-D LMMSE using the *true* per-scenario channel covariance and
                 the true noise variance. It is NOT statistics-agnostic, so it
                 is not a legal candidate; it is a near-optimal reference that
                 tells us how much room for improvement exists.

Usage (from this directory):
    python build_baseline.py [--cov-samples 4096]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
sys.path.insert(0, str(HERE))

import eval as ev  # noqa: E402
from link_config import PILOT_MASK, PILOT_SYMBOLS  # noqa: E402

COV_SEED = 50_000


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.estimate_channel


def channel_covariance(s_idx: int, num_samples: int, chunk: int = 512) -> torch.Tensor:
    """Empirical covariance E[h h^H] of the vectorized [14*72] channel for one scenario."""
    sc = ev.SCENARIOS[s_idx]
    acc = None
    n = 0
    i = 0
    while n < num_samples:
        b = min(chunk, num_samples - n)
        _, _, h = ev.generate_batch(sc, 0.0, b, seed=COV_SEED + 1000 * s_idx + i, noiseless=True)
        hv = h.reshape(b, -1).to(torch.complex128)
        c = hv.T @ hv.conj()
        acc = c if acc is None else acc + c
        n += b
        i += 1
    return acc / n


class GenieLMMSE:
    """h_hat = R_hp (R_pp + no I)^-1 h_ls_p, with the true covariance R."""

    def __init__(self, cov: torch.Tensor):
        self.cov = cov
        mask = torch.as_tensor(PILOT_MASK).reshape(-1)
        self.p_idx = torch.nonzero(mask).squeeze(-1)
        self.pilots = torch.as_tensor(PILOT_SYMBOLS).reshape(-1)[self.p_idx].to(torch.complex128)
        self.r_hp = cov[:, self.p_idx]
        self.r_pp = cov[self.p_idx][:, self.p_idx]

    def __call__(self, y: torch.Tensor, no: torch.Tensor) -> torch.Tensor:
        b = y.shape[0]
        yp = y.reshape(b, -1)[:, self.p_idx].to(torch.complex128)
        ls = yp / self.pilots                       # [B, P]
        eye = torch.eye(len(self.p_idx), dtype=torch.complex128)
        out = torch.empty(b, self.cov.shape[0], dtype=torch.complex128)
        # noise variance is constant within an evaluation batch, but stay general
        for val in torch.unique(no):
            sel = no == val
            w = self.r_hp @ torch.linalg.inv(self.r_pp + float(val) * eye)   # [N, P]
            out[sel] = ls[sel] @ w.T
        return out.reshape(y.shape).to(torch.complex64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov-samples", type=int, default=4096)
    args = ap.parse_args(argv)

    t0 = time.time()
    estimators = {
        name: _load(TASK / "baselines" / f"{name}.py")
        for name in ("ls_nearest", "ls_linear", "dft_denoise")
    }
    covs = [channel_covariance(i, args.cov_samples) for i in range(len(ev.SCENARIOS))]
    print(f"covariances estimated in {time.time() - t0:.1f}s")

    out = {"protocol": {
        "scenarios": [sc.__dict__ for sc in ev.SCENARIOS],
        "snrs_db": list(ev.SNRS_DB),
        "splits": ev.SPLITS,
        "cov_samples": args.cov_samples,
    }}
    for split in ev.SPLITS:
        res = {name: {} for name in (*estimators, "lmmse_genie")}
        with torch.no_grad():
            for s_idx, sc in enumerate(ev.SCENARIOS):
                genie = GenieLMMSE(covs[s_idx])
                for snr_idx, snr in enumerate(ev.SNRS_DB):
                    y, no, h = ev.generate_batch(sc, snr, ev.SPLITS[split]["batch_size"],
                                                 ev._point_seed(split, s_idx, snr_idx))
                    key = ev.point_key(sc, snr)
                    for name, fn in estimators.items():
                        res[name][key] = ev.mse(fn(y.clone(), no.clone()), h)
                    res["lmmse_genie"][key] = ev.mse(genie(y, no), h)
        out[split] = res
        ref = res["ls_linear"]
        for name, vals in res.items():
            nve = np.mean([vals[k] / ref[k] for k in ref])
            print(f"[{split}] {name:<12} NVE = {nve:.4f}")

    with open(HERE / ev.BASELINE_FILE, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {HERE / ev.BASELINE_FILE} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
