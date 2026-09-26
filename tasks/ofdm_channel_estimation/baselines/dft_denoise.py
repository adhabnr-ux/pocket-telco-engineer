"""
Hand-written "textbook" improvement: DFT-based (delay-domain) denoising.

1. LS estimates on the comb pilots of each pilot OFDM symbol (36 values).
2. IDFT to the delay domain. Multipath energy is concentrated in the first
   few taps (delay spread << OFDM symbol), noise is spread over all taps.
3. Keep only the taps inside a short delay window (plus a couple of
   "negative-delay" taps that catch leakage of fractional delays) and zero
   the rest; this removes most of the noise.
4. Evaluate the truncated impulse response on all 72 subcarriers
   (this also interpolates between pilots).
5. Linear interpolation in time between the two pilot symbols.

It is statistics-agnostic: it uses no knowledge of the scenario, only a
generous fixed delay window. Serves as a "competent human engineer" point of
comparison for LLM-discovered algorithms.
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
_NUM_CAUSAL_TAPS = 10     # ~4.6 us window at 463 ns/tap: covers the TDL delay profiles used here
_NUM_ANTICAUSAL_TAPS = 2


def _denoise_and_interpolate(ls: torch.Tensor, pilot_sc: torch.Tensor) -> torch.Tensor:
    """ls: [B, P] LS estimates at pilot subcarriers -> [B, FFT_SIZE]."""
    p = ls.shape[-1]
    g = torch.fft.ifft(ls, dim=-1)                       # delay taps (spacing = FFT_SIZE/P samples of the full grid)
    n = torch.arange(p)
    n_signed = torch.where(n < p // 2, n, n - p)         # wrap to signed delays
    keep = (n_signed >= -_NUM_ANTICAUSAL_TAPS) & (n_signed < _NUM_CAUSAL_TAPS)
    g = g[:, keep]
    delays = n_signed[keep].to(torch.float32)
    # H(k) = sum_n g_n exp(-j 2 pi (k - k0) n / (P * spacing)) evaluated on all subcarriers
    k = torch.arange(FFT_SIZE, dtype=torch.float32) - float(pilot_sc[0])
    phase = -2.0 * torch.pi * torch.outer(k, delays) / float(p * PILOT_SUBCARRIER_SPACING)
    basis = torch.polar(torch.ones_like(phase), phase).to(g.dtype)   # [FFT, taps]
    return g @ basis.T


def estimate_channel(y: torch.Tensor, no: torch.Tensor) -> torch.Tensor:
    pilot_sc = torch.arange(PILOT_SUBCARRIER_OFFSET, FFT_SIZE, PILOT_SUBCARRIER_SPACING)
    h_sym = torch.stack(
        [_denoise_and_interpolate(y[:, s, pilot_sc] / _PILOTS[s, pilot_sc], pilot_sc)
         for s in PILOT_OFDM_SYMBOLS],
        dim=1,
    )
    t = torch.arange(NUM_OFDM_SYMBOLS, dtype=torch.float32)
    t0, t1 = float(PILOT_OFDM_SYMBOLS[0]), float(PILOT_OFDM_SYMBOLS[-1])
    w = torch.clamp((t - t0) / (t1 - t0), 0.0, 1.0).reshape(1, -1, 1)
    return (h_sym[:, :1] * (1 - w) + h_sym[:, -1:] * w).to(torch.complex64)
