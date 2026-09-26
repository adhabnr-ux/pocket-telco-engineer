"""
Weakest baseline: least-squares at the pilots, nearest-neighbour everywhere else.
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


def estimate_channel(y: torch.Tensor, no: torch.Tensor) -> torch.Tensor:
    pilot_sc = torch.arange(PILOT_SUBCARRIER_OFFSET, FFT_SIZE, PILOT_SUBCARRIER_SPACING)
    k = torch.arange(FFT_SIZE)
    nearest_sc = torch.argmin((k[:, None] - pilot_sc[None, :]).abs(), dim=1)

    t_pilots = torch.as_tensor(PILOT_OFDM_SYMBOLS)
    t = torch.arange(NUM_OFDM_SYMBOLS)
    nearest_t = torch.argmin((t[:, None] - t_pilots[None, :]).abs(), dim=1)

    ls = torch.stack([y[:, s, pilot_sc] / _PILOTS[s, pilot_sc] for s in PILOT_OFDM_SYMBOLS], dim=1)
    ls = ls[:, :, nearest_sc]           # [B, n_pilot_sym, FFT]
    return ls[:, nearest_t, :].to(torch.complex64)
