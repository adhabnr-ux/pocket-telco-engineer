# Single-slot OFDM channel estimator: notes

**Final: `draft.py` (same as `hist/v18_final.py`). Dev NVE = 0.0352** (indoor 0.021, urban 0.055, high-speed 0.043, rural-LOS 0.021). Runtime is 0.19–0.32 s per batch-32 call, depending on machine load. I used 15 of the 40 official evaluations. Two of them failed because the code was too slow.

## What the estimator does

The estimator gets one slot: 14 × 72 resource elements (REs), 72 pilots on symbols 2 and 11 (every second subcarrier), and 16-QAM data everywhere else. It works in three stages. All of them use only `y`, `no` and `link_config`.

1. **Frequency covariance from the pilots, by Bayesian model averaging.**
   - I built a dictionary of about 255 frequency-covariance candidates Rf, each derived from a power–delay profile (PDP):
     - the public 3GPP TR 38.901 / TS 38.101-4 TDL tap tables (A–E over 32 log-spaced delay spreads from 5 to 1500 ns, plus the fixed A10/A30/B100/C60/C300/D10/D30 profiles);
     - generic shapes: uniform, exponential, and "LOS spike plus tail", with lengths from 0.25 to 32 samples.
   - Each tap's delay is spread over ±10% (plus ±0.1 sample). Without this, a candidate built from discrete taps is low-rank and fails badly when the real delay spread falls between grid points.
   - For every candidate, the exact Gaussian evidence of the 2 × 36 pilot observations is computed cheaply, using precomputed eigenvectors and a Newton search over the channel power α (with a mild Gamma prior).
   - The 8 best candidates are averaged, weighted by their evidence. This gives the per-symbol LMMSE frequency estimate at the pilot symbols and the prior Rf used by the later stages.

2. **Doppler model averaging.**
   - Candidates are Jakes time correlations (fd from 0 to 2300 Hz) plus "LOS phasor + Jakes" versions. The phasor's Doppler shift comes from the phase of the correlation between the two pilot symbols.
   - Each candidate interpolates the pilot estimates in time. It is scored by the exact 16-QAM marginal likelihood of the data REs under its prediction and prediction-error variance.
   - The final time model is a softmax average of the candidates (temperature 4), with a small penalty favouring low Doppler.

3. **Data-aided 2-D LMMSE.**
   - The prior is the Kronecker product Rt ⊗ Rf. It is solved exactly in the eigenbasis with preconditioned conjugate gradient, which handles a different noise level on every RE. The preconditioner is block-diagonal per frequency eigenvector; the same blocks give the posterior error variance of every RE.
   - Soft symbol information comes in two forms:
     - **Low SNR, or low Doppler below 14 dB:** four variational-EM steps. The weight is E|x|²/no and the pseudo-observation is y·E[x]*. Steps after the first are over-relaxed by a factor of 2, which speeds up the slow EM convergence.
     - **High SNR with high Doppler, or SNR ≥ 14 dB:** two expectation-propagation (EP) steps. Each takes exact moments of the 16-QAM/Gaussian tilted distribution given the RE's "cavity" prior. These are followed by two EM steps.
   - **High-Doppler slots (Doppler estimate ≥ 400 Hz)** first get a sequential pass. It estimates symbols one at a time, working outward from the pilots (3, 10, 1, 12, 4, 9, 0, 13, 5, 8, 6, 7). For each symbol it:
     - predicts the channel in time from the symbols already estimated;
     - computes EP messages for that symbol;
     - smooths them in frequency.

     This carries reliable decisions across the gap between the pilots in one pass. With only two pilot symbols, the gap is under-sampled at 300–500 km/h.
   - The batch is split into groups by schedule and each group is processed separately.
   - A wall-clock budget (0.36 s per call) can cut the data-aided iterations short. Every iteration gives a valid estimate, so stopping early is safe. This only matters for the batch-32 timing call; batch-1 scoring calls take about 20 ms.

**Safety measures:**
- float64 fallbacks for Hermitian inverses;
- lower bound on the EP variance;
- a check for non-finite values, falling back to the pilot-only estimate and then to LS-linear;
- single thread inside the call (with two threads, the loaded two-core machine was 3× slower);
- batch-1 results match batch-32 results.

## Why it works

- **Why DFT denoising or linear interpolation fails.** Real tap delays fall between the sampling grid points, so their energy leaks across the whole delay domain; LMMSE with a smooth (smeared) PDP covariance doesn't suffer from this leakage. Choosing the covariance by evidence lets the prior adapt to each slot: short delay spread for flat channels, long for urban ones. Averaging over the top candidates protects against picking the wrong model at low SNR.
- **Why data-aided estimation pays off.** It turns the 936 data REs into extra pilots. It gives the largest gains at high SNR and high Doppler, where interpolating in time from only two pilot symbols fails badly.
- **Why both EP and EM.** EP does not shrink toward zero, so it propagates information into uncertain regions. But at low SNR its messages are over-confident and biased. The mean-field EM step is conservative and robust at low SNR. The schedule uses each where it works.

## Weaknesses

- The remaining error in the urban and low-SNR cases comes from priors that can only be estimated from one slot. In experiments, the true per-slot frequency covariance would have cut the 0–10 dB error by a further 30–60%.
- 16-QAM decisions at 0–10 dB are unreliable. Knowing the data symbols would give about 3× less error.
- At 300–500 km/h, some slots still lock onto a rotated phase (16-QAM looks the same rotated by 90°). This happens past a deep fade that sits at a pilot symbol. A single such slot can dominate that SNR point's mean-squared error.
- The priors rely on the TDL tap tables and on Jakes Doppler. A non-TDL channel would fall back on the generic PDP shapes, which are about 5–10% worse in my tests.
- Runtime is near the middle of the allowed budget. On a heavily loaded machine the timing call is protected by the deadline, but the estimate returned for that call is less refined.

## Ideas tried, with dev NVE (1.0 = LS-linear)

| Ver. | Idea | Dev NVE |
|---|---|---|
| v1 | Sparse Bayesian learning (SBL) on pilots + time interpolation fitted to J0 (Jakes correlation) | local only; bug fixes |
| v2 | + data-aided EM with 2-D Kronecker LMMSE | **0.388** (high-speed 1.18: a phase estimate taken from the pilot correlation was spurious) |
| v3–v4 | Doppler candidates scored by data likelihood; EP messages | local; EP hurt at low SNR, helped at high SNR |
| v5 | Evidence-selected PDP shapes instead of SBL (4–7 dB better pilot estimates) + EP/EM switch at 8 dB | local 0.077; official run too slow (518 ms, **FAIL**) |
| v6b | float32, fewer candidates, faster evidence, single thread | **0.0527** (one run failed at 842 ms before the thread fix) |
| v8s | TDL-table PDP candidates + top-8 averaging + power prior; robust inverses and NaN fallback | **0.0449** |
| v10 | Smeared TDL candidates (fixes low-rank mismatch); EP→EM schedule with 4 iterations, warm-started CG | **0.0452 → 0.0418** |
| v10c | Averaging over Doppler models + low-Doppler prior | **0.0366** |
| v11/v12 | Denser delay-spread grid; over-relaxed EM; mode choice depends on Doppler; float32 fast path | **0.0356** |
| v13 | Sequential outward initialisation for high-Doppler slots (local high-speed set 0.075 → 0.058) | **0.0352** but 451 ms |
| v17 | Speed cuts + deadline | 0.0366 (190 ms) |
| **v18** | v13 accuracy + group-wise processing + deadline safeguard | **0.0352** (207–315 ms) |

**Tried and dropped:**
- EM or NNLS re-estimation of the PDP after the data-aided stage (helped the TDL-E LOS case, hurt others);
- Whittle-likelihood re-selection of the frequency model from data-aided pseudo-observations (badly worse at high SNR);
- re-weighting the frequency model average using data-aided evidence;
- separable (I/Q) EP approximation (diverged at high Doppler);
- per-slot power from all REs;
- more EM iterations;
- extra LOS K-factor candidates;
- magnitude sanity clamp;
- averaging two data-aided runs.

All tuning was done on my own Sionna TDL data (A30/B100/C300/D30/E/C-600 ns/high-Doppler A/B/C/D/E, 7 SNRs). A fresh-seed set (`syn_b`) was used as the check against overfitting. The local scores track dev closely: `syn_b` 0.040 against dev 0.035.
