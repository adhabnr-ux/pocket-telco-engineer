# Single-slot OFDM channel estimator: notes

**Final `draft.py`: dev NVE = 0.0274** (0.027373; LS-linear = 1.0). This is 17.2 dB better than LS-linear (geometric mean) and closes 1.21 of the gap to the pilot-only genie LMMSE. Runtime is about 160–175 ms per batch of 32 on one CPU thread.

Per-scenario MSE ratios on dev:

| Scenario | MSE ratio |
|---|---|
| indoor_flat | 0.0151 |
| urban_vehicular | 0.0487 |
| high_speed | 0.0308 |
| rural_los | 0.0149 |

## The algorithm in plain English

The slot is 14 OFDM symbols × 72 subcarriers. Pilots sit on the even subcarriers of symbols 2 and 11, and everything else is unknown 16QAM data. The estimator is Bayesian throughout. The channel gets a Gaussian prior with a Kronecker covariance R = R_t ⊗ R_f (time ⊗ frequency), and everything below is inference in that model.

1. **Dictionary of priors**
   * R_f comes from 102 power-delay profiles:
     * exponential PDPs with mean delay 2 ns … 1.3 µs (a factor 1.5 apart);
     * each optionally combined with a LOS spike at delay 0 (weight 0.5, 0.8 or 0.95), or with a second exponential 0.15× shorter (weight 0.3 or 0.6).
   * R_t comes from 25 Doppler spectra: Jakes (J0) and flat (sinc), with f_D from 0 to 2.3 kHz.
   * All eigendecompositions are precomputed at import time.
2. **Pilot stage (Bayesian model averaging)**
   * For every (PDP, Doppler) pair, the code computes the exact Gaussian evidence of the 72 pilot LS values. With two pilot symbols, every 2×2 time covariance has eigenvectors (1,±1)/√2, so the pilots split into a "sum" and a "difference" vector. The evidence for all 2550 pairs then reduces to two small matrix products per distinct noise level.
   * The evidences become posterior weights (temperature 0.75), and the prior is the mixture R_f = Σ w_f R_f, R_t = Σ w_d R_t. It gives the pilot-only LMMSE estimate and its per-RE variance.
   * Why it matters: at low SNR the pilots cannot tell a 30 ns channel from a 300 ns one, and a hard choice is often badly wrong. The mixture is a safe, robust prior. This gave −17 % NVE on dev.
3. **Data-aided stage (12 iterations)**
   * Each data RE gives a Gaussian "message" on its own channel coefficient (a pseudo-observation z and a precision w). The full-grid posterior mean is then recomputed with preconditioned conjugate gradients in the eigen-basis of the Kronecker prior.
   * The preconditioner is exact when the weights only vary per OFDM symbol, so 3 CG steps with a warm start are enough. It also yields per-RE posterior variances.
   * **Messages**
     * EM messages treat the soft symbol as known: z = x̄* y / E|x|², precision E|x|²/no. The soft demapper sees the extrinsic (leave-one-out) channel estimate.
     * EP messages are the moment-matched posterior of the 16QAM mixture divided by the cavity. They are used for the first 4 iterations when the slot's SNR is ≥ 6 dB, because they survive a poor initial estimate (ambiguous REs get ≈ 0 precision). After that, and always at low SNR, EM is used: EP messages turned out biased and over-confident once they settle.
   * **Progressive inclusion.** Iterations 0–1 only use data symbols next to a pilot symbol, iterations 2–3 add distance 2, 4–5 add distance 3, and then all symbols are used. This follows the channel outward from the pilots and prevents locking onto 90°-rotated constellations mid-slot when the channel moves fast. It gave large gains at high Doppler.
   * **Deterministic annealing.** The noise used by the messages is max(no, 0.1·0.7^it) for the first 8 iterations. Early decisions stay soft at high SNR, which removes most of the remaining lock-in outliers.
   * **Over-relaxed EM.** C ← C_old + 1.8·(C_new − C_old). EM converges slowly at low SNR, and this gives the benefit of several extra iterations for free.
   * **Re-estimating the hyper-parameters.** At iteration 7, when all symbols are in, the Doppler mixture weights and then the PDP mixture weights are recomputed. This uses the Gaussian evidence of the full-grid messages, with the mean precision as the noise level. The prior basis is then rebuilt. The pilots see only two time instants, so the full grid is far more informative about Doppler and delay spread (−10 % on dev).
4. **Energy constraint.** The evaluation channels have exactly unit average energy per slot. I inferred this from the fact that switching to an effective-SNR band decision left the dev result unchanged to 6 digits. So the final Gaussian posterior is tilted, h_i → h_i / (1 + g·v_i), with g found by bisection so that the expected energy is exactly 1. This is the saddle-point approximation of conditioning on ‖h‖² = N. It mostly helps low-rank (flat) channels at low SNR (−3.5 % on dev).
5. **Engineering**
   * torch complex64 with a single thread. Two threads made the wall time 5–10× worse on the loaded machine, and an evaluation failed on runtime because of it.
   * The arguments of exp are clamped at −80: exp of very negative numbers is 30× slower in CPU torch.
   * The 4-PAM posteriors use a closed form.

### Why it works

* **Low SNR.** The result is limited by the pilots and the prior. The mixture prior is close to the genie pilot-only LMMSE, and the EM data messages, the energy constraint and the Doppler/delay spread re-estimated from the full grid add 20–40 % on top.
* **Medium and high SNR.** The data symbols act as roughly 900 extra pilots, so the error falls far below the pilot-only genie. The error is often 30–300× below LS-linear at 20–25 dB, because LS-linear cannot interpolate fast channels between two pilot symbols.
* **EM check.** A Gibbs sampler (the true posterior mean under the same prior) gave the same MSE as the EM result at 5 dB. The data-aided part is effectively Bayes-optimal *given the prior*.

### Weaknesses

* At low SNR (−5 … 5 dB) for frequency-selective channels (DS ≈ 300 ns), the MSE ratio is still about 0.05–0.065, and that is where most of the remaining NVE sits.
  * A genie ensemble covariance would be about 10–20 % better. TDL profiles have fixed tap delays that generic exponential families cannot express, but gamma, delayed, truncated and two-scale exponentials, plus a per-slot SBL delay profile, recovered only 0–1 %.
  * Genie data would be about 3× better, but the Gibbs check says that gap is information-theoretic, not algorithmic.
* At very high Doppler (≥ 350 km/h) and high SNR, rare slots still lock onto a rotated solution: about 1 in 30 slots, with an MSE ratio of 0.05–0.15.
* The Kronecker prior cannot model a LOS path with its own Doppler shift (complex time correlation). Adding such families did not help on average.
* The energy constraint relies on per-slot channel normalisation, as in dev. If a test set were not normalised, that step should be switched off.
* EP messages are only usable early on. A better-calibrated EP (negative precisions, damping) could help at 5–10 dB, but my attempts failed (NaN, or worse results).

## Sequence of ideas (dev NVE from the official evaluator)

| eval | Change | dev NVE |
|---|---|---|
| 1 | Pilot LMMSE (argmax evidence over exp-PDP × Jakes) + 3 EM data-aided iterations (PCG solver) | 0.0751 |
| 2 | Hybrid EP (≥ 10 dB) / EM messages, 6 iterations, complex64 | 0.0423 |
| 3 | + progressive symbol inclusion: **failed, 2.2 s runtime** (thread contention) | — |
| 4 | Same, with a single torch thread | 0.0397 |
| 5 | Bayesian model averaging (mixture prior) at the pilot stage | 0.0329 |
| 6 | + full-grid re-estimation of the Doppler and PDP mixture weights | 0.0296 |
| 7 | EP only for the first 4 iterations; faster EP kernel; NCG = 3 | 0.0295 |
| 8 | Band chosen by effective SNR, EP threshold 6 dB (identical result, which revealed per-slot normalisation) | 0.0295 |
| 9 | Noise-floor annealing | 0.0295 |
| 10 | EM over-relaxation (1.8) | 0.0292 |
| 11 | 12 iterations | 0.0290 |
| 12 | Two-scale exponential PDP members; truncated eigen-evidence | 0.0287 |
| 13 | Unit-energy constraint (posterior tilting) | 0.0277 |
| 14 | Extrinsic channel for the EM demapper; finer delay grid (×1.5); fast pilot evidence | **0.0274** |
| 15 | Richer PDP family at the re-estimation step only (+30 ms): 0.02733, dropped as not worth it | 0.0273 |
| 16 | Per-slot SBL delay-profile refinement blended into the prior (+45 ms): 0.02731, dropped | 0.0273 |
| 17 | Final = version 14, re-run to confirm | **0.0274** |

### Tried locally and dropped

My local check used Sionna TDL-A…E, DS 10 ns–1 µs, 0–500 km/h, per-slot normalised.

* SBL on the pilots only: it over-fits.
* A soft-LS decision-directed estimator: biased and catastrophic at low SNR.
* Channel-variance gating of REs.
* Hard-decision polishing.
* Variants of the EM demapper: VB penalty, precision scaling, variance scaling.
* EP at low SNR, EP with bounded negative precisions (NaN), and EP-gated EM messages.
* Multi-hypothesis Doppler with Laplace-evidence selection or averaging: superseded by the full-grid re-estimation.
* A two-pass restart with the re-estimated prior: about 2 % for twice the cost.
* An unstructured EM covariance ("realised subspace"): it self-reinforces.
* Intensity-autocovariance statistics for the delay spread: far too noisy below 5 dB.
* Denser Doppler grid (worse, because it acts as a hyper-prior); gamma, delayed and truncated PDP shapes; higher LOS weights; LOS Doppler-shift families.
* Earlier re-estimation (the messages are too poor early on); a second re-estimation; 14 iterations (+0.5 % for +15 % time).
