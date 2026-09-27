# --- inlined reference: ls_linear ---
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


def _lin(y: torch.Tensor, no: torch.Tensor) -> torch.Tensor:
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

# --- inlined reference: dft_denoise ---
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


def _dft_core(ls: torch.Tensor, pilot_sc: torch.Tensor) -> torch.Tensor:
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


def _dft(y: torch.Tensor, no: torch.Tensor) -> torch.Tensor:
    pilot_sc = torch.arange(PILOT_SUBCARRIER_OFFSET, FFT_SIZE, PILOT_SUBCARRIER_SPACING)
    h_sym = torch.stack(
        [_dft_core(y[:, s, pilot_sc] / _PILOTS[s, pilot_sc], pilot_sc)
         for s in PILOT_OFDM_SYMBOLS],
        dim=1,
    )
    t = torch.arange(NUM_OFDM_SYMBOLS, dtype=torch.float32)
    t0, t1 = float(PILOT_OFDM_SYMBOLS[0]), float(PILOT_OFDM_SYMBOLS[-1])
    w = torch.clamp((t - t0) / (t1 - t0), 0.0, 1.0).reshape(1, -1, 1)
    return (h_sym[:, :1] * (1 - w) + h_sym[:, -1:] * w).to(torch.complex64)

# --- hybrid ---
SNR_SWITCH_DB = 12.0

def estimate_channel(y, no):
    snr_db = -10.0 * torch.log10(no.mean())
    return _dft(y, no) if snr_db < SNR_SWITCH_DB else _lin(y, no)
