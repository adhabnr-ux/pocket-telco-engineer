# Single-slot OFDM channel estimator: notes

**Final: `draft.py` (the same code as `final.py`). Official dev NVE 0.0291, runtime about 130 ms per batch of 32.**
Per scenario (dev): indoor_flat 0.0165, urban_vehicular 0.0522, high_speed 0.0317, rural_los 0.0160.
The gain over LS-linear is 16.9 dB. Fraction of the LS to genie pilot-LMMSE gap closed: 1.21, so the estimator
beats the pilot-only genie by also using the data symbols.

## What the estimator does (plain English)

Everything is computed from the one received slot `y` and the noise variance `no`. Nothing is shared across slots.

1. **Pilot stage (Bayesian model averaging, pilots only).**
   * Take the LS estimates on the 2 x 36 pilots. Rewrite them as a "sum" and a "difference" of the two pilot symbols.
     For any time correlation between symbols 2 and 11, these two components are independent.
   * Build a dictionary of generic power-delay profiles (PDPs): exponentials with delay spreads from about 9 ns to
     2 us, plus "specular spike + exponential tail" shapes. For each one the pilot-domain covariance and its
     eigen-decomposition are precomputed at import.
   * Score every hypothesis (PDP, sum-mode power u, diff-mode power v) by its exact Gaussian marginal likelihood.
     The powers u and v give the channel power P and the correlation between the two pilot symbols,
     r = (u - v)/(u + v). Inverting the Jakes law gives the Doppler, J0(2 pi fD 9T) = r.
   * Mix the PDP covariances with their posterior weights, then average the closed-form 2-D LMMSE
     interpolators (Jakes time kernel) over the posterior of (u, v). The prior is uniform in the Jakes argument,
     and hypotheses with r below the J0 minimum are excluded. This gives an estimate and an honest per-symbol error
     variance, which includes the spread between hypotheses.
2. **Sequential decision-directed tracking (SNR >= -2.5 dB).** Symbols are processed outward from the pilot
   symbols in the order {1,3,10,12} -> {0,4,9,13} -> {5,8} -> {6,7}. For each symbol:
   * Predict it with a Gaussian-process (Jakes) predictor from the symbols already estimated. The kernel mixes
     the estimated Doppler (inflated by 1.2) with a maximum-Doppler kernel (weight 0.5), so that under-estimating
     the Doppler cannot cause catastrophic over-confidence.
   * Refine the prediction with 3 inner EM iterations: soft 16-QAM decisions, then a frequency-domain LMMSE update
     around the prediction.
   This is what makes very high Doppler work at mid and high SNR: the pilots alone cannot interpolate over 9 symbols.
3. **Full-grid EM (7 iterations).**
   * **E-step:** closed-form, numerically stable 16-QAM posterior mean and second moment on every data RE.
     This gives pseudo-observations `y*conj(xbar)/E|x|^2` with weights `E|x|^2/no`, the exact EM M-step form.
   * **Covariance model averaging:** marginal-likelihood posterior weights over the PDP dictionary (76 shapes:
     exponential, uniform, two-slope exponential, spike+exp). The weights are recomputed at iterations 0 and 2.
     The same is done over 24 Jakes time-correlation candidates (0 to 2.5 kHz) at every iteration.
     The mixture covariances are eigen-decomposed. The scoring accounts for the shrinkage of soft
     pseudo-observations when SNR >= -2.5 dB.
   * **M-step:** heteroscedastic Kronecker-structured LMMSE, `h = Ut c Uf^T` with prior variance
     `lt*lf*P`. It is solved by a warm-started, diagonally preconditioned CG in the coefficient domain
     (4 steps in the first iteration, then 2).

**Why it works.**
* TDL channels are Gaussian with a separable (Kronecker) covariance: a PDP in delay and a Jakes spectrum in time.
  So LMMSE with the right covariance is essentially MMSE.
* The main problem is knowing the covariance from one slot. Scoring a small dictionary of generic, parsimonious
  models by marginal likelihood, and averaging rather than picking, is far more robust than non-parametric estimates
  (RL/SBL PDPs, sample covariances). With only 72 noisy pilots, those overfit badly.
* The data REs add about 14x more observations once the soft decisions are good. EM with soft symbols degrades
  gracefully at low SNR. The sequential tracker provides a good enough starting point at high Doppler.

**Weaknesses / limits.**
* Frequency-selective urban channels (TDL-C) are the worst case (about 0.05). The generic PDP dictionary does not match
  the discrete TDL tap structure. Substituting the true statistical covariance would give about 10-25% less error.
* At mid SNR, 16-QAM decisions are unreliable. At low SNR the result is bounded by pilot-only information.
* At very high Doppler (500 km/h) and high SNR, a few slots still fail in the sequential stage
  (error propagation when the Doppler is under-estimated from the pilots).
* The time model is pure Jakes. A LOS Doppler tone is only covered implicitly; an explicit LOS model helped only at 0 dB.
* Runtime depends on machine load: about 130 ms measured, but one submission failed at 510 ms under heavy load
  before the speed-ups.

## Sequence of ideas and results

"tune" and "val2" are my own Sionna TDL datasets (A/B/C/D/E, 10 ns to 1 us delay spread, 3 to 500 km/h).
The NVE is computed against my own LS-linear reference.

| step | idea | synthetic NVE | official dev NVE |
|---|---|---|---|
| v1 | pilot-only SBL-PDP LMMSE + Jakes-from-rho time interpolation | tune 0.66 | - |
| v2 | + hard decision-directed EM with sample-covariance time prior | 0.50 | - |
| v3-v7 | soft-symbol EM (much better at low SNR), GP sequential DD tracking for high Doppler, per-symbol error variances, SNR gating | 0.137 | - |
| v8/v9 | speed (complex64, 1 thread, PCG instead of direct Kronecker solve, Richardson-Lucy PDP instead of SBL) + PDP smoothing floor | 0.105-0.12 | **0.1008** (eval 1) |
| v10 | **Jakes time covariance selected by marginal likelihood** (instead of the sample covariance) | 0.068 | **0.0575** (eval 2) |
| v11 | exponential-PDP candidates chosen by marginal likelihood; found that the data-driven RL-PDP candidate overfits (penalised out) | 0.048 | - |
| v12 | clean rewrite: pilot stage as closed-form BMA over (u, v) + PDP; exclude hypotheses with no J0 solution | 0.0477 | **0.0336** (eval 3) |
| v12b | stable closed-form demapper (8x faster), 5 EM iterations, robust GP kernel (mix with max-Doppler Jakes) | 0.044 | **0.0322** (eval 4) |
| v13 | **Bayesian averaging of the PDP covariance** (instead of argmax) + extra PDP families + shrinkage-aware scoring | 0.0415 | eval 5 FAILED (510 ms under load) |
| v13 | speed: pilot-stage candidate subset, diagonal update in the sequential stage, PDP re-scoring only at it 0 and 2 | 0.041 | 0.0324 (eval 6) |
| v13 | time-covariance floor 1e-4 -> 1e-5, 7 EM iterations, BMA over the time covariance, warm-started PCG (2 steps/iter) | tune 0.039 / val2 0.0227 | **0.0297** (eval 7) |
| v13 | PDP BMA also in the pilot stage; shrinkage correction only at >= -2.5 dB | 0.0392 / 0.0223 | 0.0295 (eval 8) |
| final | cleaned code (identical behaviour), NTOP=2, NaN safety net | 0.0392 / 0.0223 | **0.0291** (evals 9-10) |

## Tried and rejected (no gain or worse)
* **Estimators and priors:**
  * Sample-covariance time prior, and Toeplitz ACF priors (overfit and noise-dominated).
  * SBL/RL non-parametric PDPs as the prior (overfit at low SNR; SBL too slow).
  * "Unbiased" decision-directed pseudo-observations (worse than EM-soft at low SNR).
  * A stronger low-Doppler prior.
* **EM variants:**
  * Annealed demapper noise.
  * Leave-one-out (extrinsic) channel for demapping.
  * Per-RE error variance in the demapper.
  * EM re-estimation of the power.
  * A finer delay-spread or Doppler grid.
* **Other:**
  * The uniform family alone, or a heavier family penalty.
  * An oracle LOS time model (only helps at 0 dB).
  * 10 EM iterations: about 1% better, but runtime risk.
