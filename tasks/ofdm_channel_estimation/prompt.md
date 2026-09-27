# Statistics-Agnostic OFDM Channel Estimation

## Objective

Implement an **OFDM channel estimator** that reconstructs the channel frequency response on every resource element (RE) of a 5G-NR-like slot from a handful of pilots. Your goal is to **minimize the Normalized Validation Error (NVE)**: the mean, over all evaluation points, of the ratio between your estimator's mean squared error (MSE) and the MSE of a reference least-squares estimator with linear interpolation. NVE = 1.0 means "as good as the reference"; lower is better. **The secondary goal is low runtime. Do not trade NVE for speed.**

## The link

- SISO (one transmit antenna, one receive antenna), carrier 3.5 GHz, subcarrier spacing 30 kHz.
- Resource grid: `NUM_OFDM_SYMBOLS = 14` OFDM symbols x `FFT_SIZE = 72` subcarriers.
- Pilots: OFDM symbols 2 and 11, on every other subcarrier (0, 2, 4, ..., 70) — 72 pilots in total, unit-energy QPSK. All other REs carry unknown unit-energy 16-QAM data.
- The channel has unit average energy; the noise variance `no` is known per example, so the per-RE SNR is `1 / no`.
- Evaluation SNRs: -5, 0, 5, 10, 15, 20 and 25 dB.

Everything you are allowed to know about the link is in `link_config.py` (in the workspace): `PILOT_MASK` (bool `[14, 72]`), `PILOT_SYMBOLS` (complex `[14, 72]`, zero outside pilots), `PILOT_OFDM_SYMBOLS`, `PILOT_SUBCARRIER_SPACING`, `FFT_SIZE`, `NUM_OFDM_SYMBOLS`, `SUBCARRIER_SPACING`, `CARRIER_FREQUENCY`, `OFDM_SYMBOL_DURATION`, and more. Import it with `from link_config import ...`.

## The channel is unknown

Each batch comes from one of several 3GPP TR 38.901 TDL propagation scenarios, ranging from nearly flat and static indoor channels to highly frequency-selective urban channels and fast-moving (high-Doppler) users, both line-of-sight and non-line-of-sight. **Your estimator is not told which scenario produced a batch**, nor its delay spread or Doppler. All examples in one call share the same scenario and SNR. A good estimator must adapt to the channel it observes, and must never be much worse than the reference on any scenario: the metric averages *ratios*, so a catastrophic failure on one scenario costs more than a modest gain elsewhere.

## Interface

Create `draft.py` defining:

```python
def estimate_channel(y, no):
    """
    y  : torch.complex64 tensor [batch, 14, 72]  received resource grid
    no : torch.float32   tensor [batch]          noise variance per example
    returns: complex tensor [batch, 14, 72]      channel estimate on every RE
    """
```

A numpy array is also accepted as the return value.

Available packages: `torch` (CPU), `numpy`, `scipy`, and Sionna 2.x (PyTorch-based; its modules live under `sionna.phy`, e.g. `from sionna.phy.ofdm import ...`; the old `sionna.ofdm` paths no longer exist). Nothing else is installed (no scikit-learn, no TensorFlow) and there is no internet access.

## Evaluation

- Call the evaluation tool (or run `python eval.py draft.py`). The first output line is `SUCCESS, <NVE>, <runtime_s>` or `FAILURE,` followed by an error message. Additional lines give per-scenario MSE ratios, the gain in dB, and the fraction of the gap closed toward a genie estimator: the pilot-only LMMSE estimator given the true channel statistics (a reference you cannot use directly; estimators that also exploit the data symbols can beat it, giving a fraction above 1).
- Runtime is the average time of one call on a batch of 32 grids. Calls slower than 500 ms fail.
- Returning NaN/Inf, a wrong shape, or a real-valued tensor fails.

## Rules

- `estimate_channel` may only use `y`, `no` and the contents of `link_config.py`. Estimating statistics from the received data itself (e.g. across the batch, across subcarriers, or across time) is allowed and encouraged.
- No introspection or access to evaluation internals: source files containing `inspect`, `_getframe`, `f_back`, `f_locals`, `gc.`, `sys.modules`, `settrace`, or references to `eval.py`/`baseline_mse` are rejected.
- Final results are reported on a **held-out split** generated with different random seeds. Do not overfit to the evaluation draws.
- No pre-trained weights downloaded from the internet. Anything you fit must be computed inside `estimate_channel` or at import time from `link_config.py` alone.

## Hints

Consider exploiting the structure of multipath channels (sparse, short delay profiles relative to the OFDM symbol), noise-aware shrinkage, correlation across frequency and time estimated from the data, robust fallbacks when an assumption fails, and SNR-dependent behaviour. The classic DFT-denoising estimator, for example, is excellent on some scenarios and poor on others at high SNR: understanding why is a good place to start.
