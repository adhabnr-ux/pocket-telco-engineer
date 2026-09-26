"""
Reference baseline: least-squares (LS) estimates at the pilots, then linear
interpolation across frequency and time.

This is the normalizer of the task metric: its NVE is exactly 1.0.

Frequency: pilots sit on every other subcarrier; values in between are the
average of the two neighbours; the last (odd) subcarrier past the final
pilot is held at the nearest pilot value.
Time: linear interpolation between the two pilot OFDM symbols; symbols
before the first / after the last pilot symbol hold the nearest pilot value.
"""

import torch

from link_config import (
    FFT_SIZE,
    NUM_OFDM_SYMBOLS,
    PILOT_OFDM_SYMBOLS,
    PILOT_SUBCARRIER_OFFSET,
    PILOT_SUBCARRIER_SPACING,
    PILOT_SYMBOLS,
)

_PILOTS = torch.as_tensor(PILOT_SYMBOLS)


def _interp_frequency(h_p: torch.Tensor, pilot_sc: torch.Tensor) -> torch.Tensor:
    """h_p: [..., num_pilot_sc] -> [..., FFT_SIZE] via linear interpolation (flat extrapolation)."""
    k = torch.arange(FFT_SIZE, dtype=torch.float32)
    pos = pilot_sc.to(torch.float32)
    idx = torch.clamp(torch.searchsorted(pos, k, right=True) - 1, 0, len(pos) - 2)
    x0, x1 = pos[idx], pos[idx + 1]
    w = torch.clamp((k - x0) / (x1 - x0), 0.0, 1.0)
    return h_p[..., idx] * (1 - w) + h_p[..., idx + 1] * w


def estimate_channel(y: torch.Tensor, no: torch.Tensor) -> torch.Tensor:
    pilot_sc = torch.arange(PILOT_SUBCARRIER_OFFSET, FFT_SIZE, PILOT_SUBCARRIER_SPACING)
    t_pilots = list(PILOT_OFDM_SYMBOLS)

    # LS on each pilot symbol, then frequency interpolation: [B, n_pilot_symbols, FFT]
    h_sym = []
    for s in t_pilots:
        ls = y[:, s, pilot_sc] / _PILOTS[s, pilot_sc]
        h_sym.append(_interp_frequency(ls, pilot_sc))
    h_sym = torch.stack(h_sym, dim=1)

    # Time interpolation
    t = torch.arange(NUM_OFDM_SYMBOLS, dtype=torch.float32)
    t0, t1 = float(t_pilots[0]), float(t_pilots[-1])
    w = torch.clamp((t - t0) / (t1 - t0), 0.0, 1.0).reshape(1, -1, 1)
    h_hat = h_sym[:, :1] * (1 - w) + h_sym[:, -1:] * w
    return h_hat.to(torch.complex64)
