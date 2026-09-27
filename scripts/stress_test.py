"""Out-of-distribution stress test: do discovered estimators survive channels they were not tuned on?

Every candidate is called one slot at a time (the realistic setting) on channels that differ from
the benchmark in one controlled way each:

  tdl_e_los          TDL-E (a LOS profile absent from the benchmark), 50 ns, 0-10 m/s
  very_long_delay    TDL-C with a 1 us delay spread (3x the benchmark's largest)
  unnormalized       TDL-A 100 ns, 0-30 m/s, WITHOUT per-slot energy normalization
                     (real fading: each slot's energy varies; the benchmark normalizes every slot)
  random_multipath   not 3GPP at all: 2-12 paths with random continuous delays (0-2 us),
                     random powers and per-path Doppler (up to 60 m/s at 3.5 GHz)

    python scripts/stress_test.py name=path/to/estimator.py [name=...]

Prints and writes results/stress_test.{json,md}: NVE vs LS-linear per scenario (lower is better) and
the pilot-only genie LMMSE for reference.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "tasks" / "ofdm_channel_estimation" / "eval"
sys.path.insert(0, str(EVAL))

import eval as ev  # noqa: E402
from build_baseline import GenieLMMSE  # noqa: E402
from link_config import (CARRIER_FREQUENCY, DATA_NUM_BITS_PER_SYMBOL, FFT_SIZE, NUM_OFDM_SYMBOLS,  # noqa: E402
                         OFDM_SYMBOL_DURATION, PILOT_MASK, PILOT_SYMBOLS, SUBCARRIER_SPACING)

SNRS = (0.0, 10.0, 20.0)
BATCH = 32
SEED = 777_000


def _qam_grid(b: int, gen: torch.Generator) -> torch.Tensor:
    from sionna.phy.mapping import QAMSource

    mask = torch.as_tensor(PILOT_MASK)
    x = torch.as_tensor(PILOT_SYMBOLS).expand(b, -1, -1).clone()
    x[:, ~mask] = QAMSource(DATA_NUM_BITS_PER_SYMBOL)([b, int((~mask).sum())]).to(x.dtype)
    return x


def _tdl(model, ds, vmin, vmax, normalize):
    def gen(b, seed):
        import sionna.phy
        from sionna.phy.channel import OFDMChannel
        from sionna.phy.channel.tr38901 import TDL
        from sionna.phy.ofdm import ResourceGrid

        sionna.phy.config.seed = seed
        torch.manual_seed(seed)
        rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS, fft_size=FFT_SIZE,
                          subcarrier_spacing=SUBCARRIER_SPACING, cyclic_prefix_length=6)
        ch = OFDMChannel(TDL(model, ds, CARRIER_FREQUENCY, min_speed=vmin, max_speed=vmax), rg,
                         add_awgn=False, normalize_channel=normalize, return_channel=True)
        _, h = ch(torch.zeros(b, 1, 1, NUM_OFDM_SYMBOLS, FFT_SIZE, dtype=torch.complex64))
        return h.reshape(b, NUM_OFDM_SYMBOLS, FFT_SIZE).to(torch.complex64)
    return gen


def _random_multipath(b, seed):
    rng = np.random.default_rng(seed)
    k = np.arange(FFT_SIZE)[None, :]
    t = np.arange(NUM_OFDM_SYMBOLS)[:, None] * OFDM_SYMBOL_DURATION
    fd_max = 60.0 / 3e8 * CARRIER_FREQUENCY
    out = np.zeros((b, NUM_OFDM_SYMBOLS, FFT_SIZE), dtype=np.complex128)
    for i in range(b):
        n = rng.integers(2, 13)
        tau = np.sort(rng.uniform(0, 2e-6, n))
        p = rng.exponential(1.0, n) * np.exp(-tau / rng.uniform(0.2e-6, 1e-6))
        p /= p.sum()
        a = np.sqrt(p / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        fd = fd_max * np.cos(rng.uniform(0, 2 * np.pi, n))
        for l in range(n):
            out[i] += a[l] * np.exp(2j * np.pi * fd[l] * t) * np.exp(-2j * np.pi * k * SUBCARRIER_SPACING * tau[l])
    return torch.as_tensor(out.astype(np.complex64))


SCENARIOS = {
    "tdl_e_los": _tdl("E", 50e-9, 0.0, 10.0, True),
    "very_long_delay": _tdl("C", 1000e-9, 0.0, 3.0, True),
    "unnormalized": _tdl("A", 100e-9, 0.0, 30.0, False),
    "random_multipath": _random_multipath,
}


def observe(h: torch.Tensor, snr_db: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    b = h.shape[0]
    torch.manual_seed(seed)
    x = _qam_grid(b, g)
    no = torch.full([b], 10 ** (-snr_db / 10), dtype=torch.float32)
    noise = (torch.randn(h.shape, generator=g) + 1j * torch.randn(h.shape, generator=g)) * np.sqrt(no[0] / 2)
    return (h * x + noise.to(h.dtype)).to(torch.complex64), no


def load(path: Path):
    spec = importlib.util.spec_from_file_location("cand_" + path.stem + str(abs(hash(str(path)))), path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod.estimate_channel


def main(argv: list[str]) -> int:
    cands = {"ls_linear": REPO / "tasks/ofdm_channel_estimation/baselines/ls_linear.py",
             "dft_denoise": REPO / "tasks/ofdm_channel_estimation/baselines/dft_denoise.py"}
    for a in argv:
        name, p = a.split("=", 1)
        cands[name] = Path(p).resolve()
    fns = {n: load(p) for n, p in cands.items()}

    data = {}
    for s_idx, (sname, gen) in enumerate(SCENARIOS.items()):
        cov_h = torch.cat([gen(256, SEED + 50_000 + 100 * s_idx + j) for j in range(8)]).reshape(2048, -1)
        cov = (cov_h.to(torch.complex128).T @ cov_h.to(torch.complex128).conj()) / cov_h.shape[0]
        data[sname] = (GenieLMMSE(cov), [(snr, gen(BATCH, SEED + 100 * s_idx + i)) for i, snr in enumerate(SNRS)])

    mse = {n: {} for n in (*fns, "lmmse_genie")}
    times = {n: 0.0 for n in fns}
    with torch.no_grad():
        for s_idx, (sname, (genie, pts)) in enumerate(data.items()):
            for i, (snr, h) in enumerate(pts):
                y, no = observe(h, snr, SEED + 9_000 + 100 * s_idx + i)
                key = f"{sname}@{snr:g}dB"
                mse["lmmse_genie"][key] = ev.mse(genie(y, no), h)
                for n, fn in fns.items():
                    t0 = time.perf_counter()
                    try:
                        est = torch.cat([torch.as_tensor(fn(y[j:j + 1].clone(), no[j:j + 1].clone()))
                                         for j in range(BATCH)])
                        m = ev.mse(est.to(torch.complex64), h)
                        m = m if np.isfinite(m) else float("inf")
                    except Exception as exc:  # a crash on an unseen channel is a result too
                        print(f"  {n} crashed on {key}: {type(exc).__name__}: {exc}")
                        m = float("inf")
                    times[n] += time.perf_counter() - t0
                    mse[n][key] = m
            print(f"done {sname}", flush=True)

    ref = mse["ls_linear"]
    rows = []
    for n, vals in mse.items():
        per = {s: float(np.mean([vals[k] / ref[k] for k in ref if k.startswith(s + "@")])) for s in SCENARIOS}
        rows.append({"estimator": n, "nve_ood": float(np.mean([vals[k] / ref[k] for k in ref])), **per})
    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "stress_test.json").write_text(json.dumps({"rows": rows, "mse": mse, "snrs": SNRS,
                                                           "batch": BATCH}, indent=1))
    hdr = "| estimator | OOD NVE | " + " | ".join(SCENARIOS) + " |"
    lines = [hdr, "|" + "---|" * (len(SCENARIOS) + 2)]
    for r in sorted(rows, key=lambda r: r["nve_ood"]):
        lines.append(f"| {r['estimator']} | {r['nve_ood']:.4f} | " + " | ".join(f"{r[s]:.4f}" for s in SCENARIOS) + " |")
    (out_dir / "stress_test.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
