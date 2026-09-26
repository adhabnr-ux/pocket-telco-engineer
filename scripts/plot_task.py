"""Figures for understanding the task (and for your poster / write-up).

    python scripts/plot_task.py                       # channels + reference MSE curves
    python scripts/plot_task.py --candidate runs/x/candidates/g01_w02_s03.py --label "Discovered"

Writes figures/channels.png and figures/mse_vs_snr.png.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TASK = REPO / "tasks" / "ofdm_channel_estimation"
sys.path.insert(0, str(TASK / "eval"))

import eval as ev  # noqa: E402
from build_baseline import GenieLMMSE, channel_covariance  # noqa: E402


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.estimate_channel


def plot_channels(out: Path) -> None:
    fig, axes = plt.subplots(1, len(ev.SCENARIOS), figsize=(3.3 * len(ev.SCENARIOS), 3.1), sharey=True)
    for ax, (i, sc) in zip(axes, enumerate(ev.SCENARIOS)):
        _, _, h = ev.generate_batch(sc, 30.0, 1, seed=7 + i, noiseless=True)
        im = ax.imshow(20 * np.log10(h[0].abs().numpy() + 1e-6), aspect="auto", origin="lower",
                       cmap="viridis", vmin=-25, vmax=8)
        ax.set_title(f"{sc.name}\nTDL-{sc.model}, {sc.delay_spread * 1e9:.0f} ns, "
                     f"{sc.min_speed:.0f}-{sc.max_speed:.0f} m/s", fontsize=8)
        ax.set_xlabel("subcarrier")
    axes[0].set_ylabel("OFDM symbol")
    fig.colorbar(im, ax=axes, label="|H| [dB]", shrink=0.85)
    fig.suptitle("One channel realization per scenario (what the estimator must reconstruct)", fontsize=10, y=1.1)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_mse(out: Path, candidate: Path | None, label: str, split: str) -> None:
    ests = {name: _load(TASK / "baselines" / f"{name}.py") for name in ("ls_nearest", "ls_linear", "dft_denoise")}
    if candidate:
        ests[label] = _load(candidate)
    styles = {"ls_nearest": ("0.6", ":"), "ls_linear": ("0.25", "--"), "dft_denoise": ("#7c3aed", "-."),
              "lmmse_genie": ("black", "-"), label: ("#dc2626", "-")}
    fig, axes = plt.subplots(1, len(ev.SCENARIOS), figsize=(3.4 * len(ev.SCENARIOS), 3.3), sharey=True)
    bs = ev.SPLITS[split]["batch_size"]
    for s_idx, (ax, sc) in enumerate(zip(axes, ev.SCENARIOS)):
        genie = GenieLMMSE(channel_covariance(s_idx, 2048))
        curves = {k: [] for k in (*ests, "lmmse_genie")}
        with torch.no_grad():
            for snr_idx, snr in enumerate(ev.SNRS_DB):
                y, no, h = ev.generate_batch(sc, snr, bs, ev._point_seed(split, s_idx, snr_idx))
                for k, fn in ests.items():
                    curves[k].append(ev.mse(torch.as_tensor(fn(y.clone(), no.clone())), h))
                curves["lmmse_genie"].append(ev.mse(genie(y, no), h))
        for k, v in curves.items():
            c, ls = styles.get(k, ("#0ea5e9", "-"))
            ax.semilogy(ev.SNRS_DB, v, ls=ls, color=c, lw=2 if k in (label, "lmmse_genie") else 1.4,
                        marker="o", ms=3, label=k)
        ax.set_title(sc.name, fontsize=9)
        ax.set_xlabel("SNR [dB]")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("channel MSE")
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"Channel estimation error per scenario ({split} split)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, default=None)
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--split", default="test", choices=list(ev.SPLITS))
    ap.add_argument("--out-dir", type=Path, default=REPO / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_channels(args.out_dir / "channels.png")
    plot_mse(args.out_dir / "mse_vs_snr.png", args.candidate, args.label, args.split)
    print(f"wrote {args.out_dir / 'channels.png'} and {args.out_dir / 'mse_vs_snr.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
