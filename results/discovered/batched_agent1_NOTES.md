# OFDM channel estimator: notes

Best dev NVE: **0.03124** (evaluation 11 of 40 used; `draft.py` is the same file as `.history/eval11.py`).
Per scenario: indoor_flat 0.020, urban_vehicular 0.053, high_speed 0.032, rural_los 0.020.
Runtime is about 90-130 ms per batch of 32. The machine was shared and loaded, so timings varied.
The pilot-only genie LMMSE scores about 0.15 on this metric. We beat it because we also use the unknown 16-QAM data symbols.

## The algorithm

1. **Delay profile from the batch.** All 32 examples in a call share one scenario. Least-squares estimates at the 72 pilots give a frequency autocorrelation. From it we fit a power-delay profile (PDP) on a 0.25-Ts grid over [-0.5, 24] Ts.
   - Below 4 dB SNR: non-negative least-squares covariance matching.
   - At 4 dB and above: multiple-snapshot sparse Bayesian learning (MacKay updates, 50 iterations). SBL accounts for the noise and gives much cleaner profiles at high SNR.
   - Keeping the grid short is important. Letting it reach the comb-2 aliasing region, up to 36 Ts, broke interpolation to the odd subcarriers.
2. **Time correlation from the two pilot symbols.**
   - The normalised correlation between symbols 2 and 11 is measured inside the pilot signal subspace, which removes most of the noise.
   - It is then mapped to a Jakes J0 model. The mapping allows negative correlation, i.e. past the first zero of J0.
   - If |c| > 0.8, we also apply a common phase rotation (Doppler-shifted LOS).
3. **Pilot-only 2-D LMMSE** with the Kronecker prior Rt ⊗ Rf. This is the starting point and gives the per-RE error variance.
4. **Data-aided variational-Bayes EM.** This is where the large gain comes from.
   - **E-step:** exact 4-PAM posteriors (separately for I and Q) of each 16-QAM data symbol, given the current channel mean and variance. A leave-one-out cavity removes the RE's own contribution, so an RE does not confirm itself.
   - **M-step:** each RE becomes a Gaussian pseudo-observation `z = y·x̄*/E|x|²` with precision `E|x|²/no`. The weighted LMMSE is solved in the reduced Kronecker eigen-subspace (at most 200 dims) by preconditioned CG, warm-started from the previous iteration.
   - **Fast channels** (low pilot correlation) are processed progressively outward from the pilot symbols, 2 iterations per distance ring. During this phase the noise is floored at an 8 dB SNR (annealing), which stops early hard decisions from locking in phase slips.
5. **EM re-estimation of the statistics.** Rt and Rf are replaced by the posterior second moments over the whole grid: `ĥĥᴴ` plus an approximate posterior covariance. This is 14× more data than the pilots and needs no model, so it handles LOS rotation and arbitrary Doppler spectra. We run 4 cycles, each followed by 2 VB iterations.
6. **Safety.** NNLS has a fallback, NaNs are replaced, and time guards stop the iterations at 0.25 s / 0.3 s.

**Why it works.**
- The statistics are learned per batch, so the estimator adapts to the scenario.
- The Kronecker eigen-subspace concentrates the channel into a few dimensions, which lets us average away noise.
- The data REs turn 72 pilots into roughly 1008 soft observations. This matters most for high Doppler, where 2 pilot symbols cannot follow the channel, and at high SNR.

**Weak points.**
- Low SNR (-5 to 5 dB) on frequency-selective channels is the weakest area, with ratios of 0.05-0.1. Data decisions carry little information there. Tests with known data show roughly 5× lower error is theoretically available, but VB and a quick EP variant could not extract it.
- LOS channels at very high speed and low SNR: the Kronecker model is only approximately right when a Doppler-shifted LOS path is mixed with diffuse taps.
- Speeds above about 600 km/h at 3.5 GHz: the J0 inversion from two pilots is ambiguous. A fourth-moment Doppler estimate helped on synthetic data but hurt dev high_speed, so it is off.
- Delay spreads with taps beyond about 11 µs: an adaptive grid extension exists but is disabled (`long_snr=99`), because it triggered slightly on dev urban.
- The runtime guard makes the results depend a little on machine load.

## Sequence of ideas (dev NVE from official evaluations; other numbers are from my synthetic Sionna TDL benchmark)

| # | Idea | dev NVE |
|---|------|---------|
| 1 | Pilot-only LMMSE: NNLS PDP (grid limited to 24 Ts), J0 time model with sign fix | 0.2115 |
| - | (sim) hard-decision DD: big high-SNR gain, but mid-SNR loss and catastrophes | - |
| 2 | Soft VB-EM data-aided iterations, progressive schedule, rotation threshold 0.8, reduced-subspace direct solver | 0.0469 |
| 3 | Lower PDP floor, subspace-projected pilot correlation, SBL PDP above 4 dB, faster torch code | 0.0382 |
| 4 | EM re-estimation of Rt (and NNLS-fitted Rf below 8 dB), 2 cycles | 0.0346 |
| 5 | EM Rf taken directly as the raw posterior second moment (no PDP refit) at all SNRs | 0.0324 |
| 6 | 4 EM cycles, fewer pre-EM iterations, torch-only EM | 0.0313 |
| 7 | Leave-one-out E-step and SNR-capped annealing in the progressive phase (sim: C300 at 500 km/h, 25 dB: 0.07 → 0.002) | 0.0312 |
| 8 | Fourth-moment (|y|² correlation) Doppler estimate for the fast regime: hurt high_speed | 0.0414 (reverted) |
| 9-10 | Adaptive long delay grid (cleaned-up final code) | 0.0314 |
| 11 | Same, with the long-grid extension disabled | **0.0312** |

Tried on synthetic data and rejected (no gain, or worse):
- Empirical Toeplitz Rt/Rf with PSD clipping: small eigenvalues get noisy.
- NNLS Doppler-spectrum fit from pseudo-observations.
- Expectation propagation sites.
- Tempered data likelihood.
- Inflating the error variance in the E-step.
- More than 4 EM cycles.
- Larger or smaller subspace thresholds.
- A delay grid with tmax of 20 or 28 Ts.

The synthetic benchmark covered 10 TDL scenarios × 7 SNRs, using my own LS-linear reference. The final version scores 0.0415 there, against 0.20 for a separable genie pilot LMMSE.
