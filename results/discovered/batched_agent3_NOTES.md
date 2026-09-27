# Semi-blind, statistics-adaptive OFDM channel estimator (ws3)

Best dev NVE: **0.0526** (evaluation 10, file `draft.py`; runtime about 105 ms per batch of 32).
Official evaluations used: 11 of 40.

## What the algorithm does

1. **Pilot least squares (LS).** Compute LS channel estimates on the 72 pilots (symbols 2 and 11, every second subcarrier).

2. **Frequency statistics (power-delay profile, PDP).**
   - Average the autocorrelation of the pilot LS vectors over the batch: 64 snapshots of 36 pilots, at lags 0 to 35, with the noise variance `no` removed at lag 0.
   - Fit a non-negative PDP on a fine delay grid (from -2 to 24 samples, step 1/8 sample) with non-negative least squares (NNLS).
   - For each 1-sample delay bin, subtract twice that bin's noise standard deviation, `(no/36)/sqrt(64)`, and floor the result at zero (soft threshold). This removes the spurious far-delay taps that noise creates at low SNR, and was worth 10 to 15% at -5 to 5 dB.
   - Above 7 dB, smooth the PDP by ±0.25 sample. Above 22 dB, add a 1e-3 floor over the delay support. Without these, the prior is over-confident about the fine delay structure, and at high SNR the small-eigenvalue tail of the prior decides the error floor.

3. **Time statistics.**
   - Measure the complex pilot correlation rho between symbols 2 and 11 (9 symbols apart).
   - Invert J0 so that the Jakes model J0(x·d/9) matches |rho|.
   - When the phase of rho is beyond ±90°, decide between two cases:
     - NLOS fast fading, where J0 has gone negative: flip the sign.
     - LOS Doppler rotation: keep the phase.
   - The decision uses |rho| and an "envelope fading depth", cov(|y_t|², |y_t+1|²)/P0². This is about 1 for Rayleigh fading and about 0.2 for strong LOS.
   - The phase of rho is kept only when |rho| is large enough to be reliable.

4. **Joint 2-D MAP/LMMSE estimator.**
   - Build the prior R = Rt ⊗ Rf and work in its Kronecker eigenbasis. Keep only product eigenvalues above 1e-5 of the maximum, which gives K ≤ 200 and usually 10 to 80 columns.
   - Every resource element (RE) is an observation with its own precision d (1/no for pilots).
   - The posterior needs Uᴴ D U. The Kronecker structure computes it with two small matrix multiplies instead of a 1008-dimensional product. It then needs one K×K Cholesky factorisation per example.

5. **Semi-blind expectation propagation (EP) on the 16-QAM data REs.**
   - For each data RE, take the cavity = posterior ÷ the RE's own message.
   - Combine the cavity with the exact 16-component likelihood Σₓ p(x) CN(y; h·x, no).
   - Moment-match the resulting h posterior. The implementation is vectorised by grouping the three |x|² classes.
   - The extrinsic message becomes a "virtual pilot" with precision d.
   - Messages are tempered by alpha = clip((SNR_dB − 2)/10, 0, 0.75).
   - Iterate up to 8 times, stopping early when the change is below 0.01·no.
   - Below 2 dB, apply a single EP step with alpha = 0.05·SNR_linear.

6. **EM re-estimation of the Doppler spectrum** (SNR ≥ 3 dB).
   - Estimate the time autocorrelation from the posterior channel plus its posterior covariance.
   - Fit it by NNLS on a Doppler grid of ±0.12 cycles per symbol. This fits Jakes spectra, LOS lines, and the realised shape of the batch.
   - Run the semi-blind pass again from a cold start (6 iterations) with the new time prior.

7. **Safety checks.**
   - Jittered Cholesky if a factorisation fails.
   - PDP fallback when the fit is empty.
   - Fall back to the pilot-only estimate if the refinement returns non-finite values or blows up.
   - Torch runs single-threaded inside the call. Under a loaded machine, thread contention made it 5 to 7× slower.

## Why it works

- The channel lives in a low-dimensional subspace (short delay spread, limited Doppler). With second-order statistics estimated from the batch, a 2-D LMMSE on the pilots reaches close to genie performance at low SNR.
- At medium and high SNR the reference estimator is limited by the pilots. The 936 data REs carry much more information than the 72 pilots. EP uses them without hard decisions and with calibrated confidence, so the method beats the "genie" pilot-only LMMSE by a wide margin (the dev gap-closed fraction is 1.17).
- The EM Doppler re-estimation fixes the cases where one pilot lag cannot identify the time correlation: high Doppler, LOS rotation, and realised correlations that differ from J0.

## Where it is weak

- **Medium SNR (5 to 10 dB).** The 16-QAM symbol error rate is about 30% there even with a perfect channel. EP is still 2 to 3× worse than the known-data bound. An EP whose cavity is the true channel would be about 35% better, so part of the gap is convergence to a sub-optimal fixed point.
- **Very high Doppler (about 500 km/h) at high SNR.** Occasionally one example locks into a wrong solution in the middle of the slot (a 16-QAM rotational ambiguity far from both pilot symbols).
- **LOS channels at high SNR.** The Kronecker prior assumes every delay has the same Doppler spectrum, which is false when a LOS tap exists. A rank-1 LOS augmentation helped rural LOS at 25 dB but broke other cases, so it was dropped.
- **Low SNR (-5 and 0 dB).** Pilot-only statistics limit it: about 5 to 10% worse than a Toeplitz oracle covariance.

## Sequence of ideas and dev NVE (official evaluations)

| # | Change | dev NVE |
|---|--------|---------|
| 1 | Pilot-only 2-D LMMSE: NNLS PDP (0.25-sample grid) + J0 from rho, with the sign fix | 0.2888 |
| 2 | + EP decision-directed refinement (numpy, multi-threaded) | FAILED: 903 ms per call (thread contention) |
| 3 | Same in torch, single-threaded; Kronecker precision assembly | 0.0762 |
| 4 | Faster EP, early stopping, Doppler re-estimation for fast channels | 0.0658 |
| 5 | Noise-aware PDP bin threshold, gated rho phase, finer delay grid with negative delays, SNR-dependent PDP smoothing and floor | 0.0577 |
| 6 | 8 + 8 EP iterations, Doppler re-estimation always on (SNR ≥ 3 dB) | 0.0536 |
| 7 | LOS-rotation versus J0-negative disambiguation (large local gain; no change on dev) | 0.0536 |
| 8 | Alpha schedule (A1 = 10, AMAX = 1) + robustness guards | 0.0542 |
| 9 | AMAX = 0.75 + one weak EP step at low SNR | 0.0526 |
| 10 | Code clean-up and refactor (final) | **0.0526** (0.052581) |
| 11 | Finer rank truncation (EPS = 1e-6): no gain and slower, reverted | 0.0528 |

## Ideas tried only locally (self-generated Sionna TDL data) and rejected

These are relative to the method at the time they were tried:

- **Separable two-stage (frequency then time) smoothing with heteroscedastic noise.** About 30% worse than the joint 2-D solution.
- **Hard-decision decision-directed estimation with confidence gating.** Selection bias; worse than EP at every SNR.
- **EM with soft symbols (point channel estimate).** Biased at low SNR.
- **Untempered EP (alpha = 1).** Its extrinsic messages carry a radial bias at low SNR (they inflate |h|), which is why alpha is tempered and EP is essentially off below 2 dB.
- **Gating EP messages by symbol confidence.** Much worse because of selection bias.
- **Damped EP, alpha ramp-up, per-example posterior variance.** No gain.
- **Maximum-likelihood (EM fixed-point) PDP estimation.** No better than NNLS.
- **Smoothness-regularised NNLS.** Helped nothing and hurt flat channels.
- **Frequency-PDP re-estimation from the posterior.** Neutral or worse (tail mismatch).
- **Envelope-correlation Doppler estimation**, and a 2-parameter J0/cos family fitted to it. No gain once the sign and LOS logic was fixed; the Toeplitz-oracle time covariance gave no further gain for NLOS.
- **Rank-1 LOS augmentation (non-separable prior).** Helped some LOS cases at 25 dB and hurt others; not robust.
