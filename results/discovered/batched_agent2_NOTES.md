# OFDM channel estimator (ws2): notes

Final dev NVE: **0.0281** (indoor_flat 0.0177, urban_vehicular 0.0479, high_speed 0.0299, rural_los 0.0170).
Runtime is about 130 ms per batch of 32. I used 11 of the 40 official evaluations. Nearly all design and tuning
ran on my own Sionna TDL data (A–E, delay spreads from 10 ns to 1 µs, speeds from 3 to 500 km/h, all 7 SNRs, two independent sets with 24 scenarios in total).

## What the algorithm does

The estimator is an LMMSE / EM estimator that learns the channel statistics from the batch it is given. It uses the pilots and also the 16-QAM data REs as soft pilots. All examples in a call share one set of statistics.

1. **Delay profile (frequency covariance).** Compute LS estimates on the 2×36 pilots. The 64 pilot vectors are
   snapshots for M-SBL (sparse Bayesian learning with MacKay updates and known noise). M-SBL fits a power-delay profile on a
   0.25-sample delay grid from -3 to 30 samples. Rf = A·diag(γ)·Aᴴ is Toeplitz and PSD by construction, and it
   extends from the even pilot subcarriers to all 72. Because SBL is a maximum-likelihood method with the known noise level, it gives a sparse profile. That
   matters a lot at high SNR, where spurious small components pass noise. Earlier NNLS fits of the autocorrelation
   were much worse there.
2. **Doppler (time covariance).** The pilot symbols are 9 symbols apart, so the only direct measurement is the lag-9
   correlation c. It is inverted through the Jakes model J0(x) over [0, 3.83], so negative correlations up to about 585 km/h are handled.
   If |c| is large and its phase is not near ±π, the phase is kept as a LOS Doppler shift.
3. **Two-group model (non-separable channels).** The delay grid points within ±0.5 sample of the strongest delay
   form group 1; the rest form group 2. Each group gets its own time covariance, estimated from Wiener-separated
   pilots. The prior is R = Rt₁⊗Rf₁ + Rt₂⊗Rf₂. This captures LOS channels, where the LOS tap rotates while the
   diffuse taps fade (Jakes). It also helped NLOS channels, with a small loss only on very flat indoor channels.
4. **Solver.** The channel is represented in the truncated eigenbases (Ut⊗Uf) of each group. Components are kept if
   their eigenvalue exceeds 0.6·noise, which works as regularization. The posterior mean is solved per example with
   the Kronecker structure (cost is about r³ with r ≤ 200, not 1008³). The per-RE posterior variance is computed too.
5. **Data-aided EM.** Each step does soft 16-QAM demapping (separable 4-PAM, with posterior variance inflated ×3 because the model
   is over-confident). Then an EM M-step uses weight E|x|²/n0 and data x̄*y/n0. This is proper EM. The "soft-pilot"
   alternative, weight |x̄|²/(n0+|h|²v), is biased by self-reinforcement and failed badly at low SNR. Details:
   * **Progressive ring schedule.** Data symbols are admitted in rings by distance from the pilot symbols (1,3,10,12 →
     4,9 → …). The extrapolated edge symbols come in half a ring later, and rings grow at half speed when the pilot correlation
     indicates fast fading. This avoids committing early to wrong decisions in the poorly predicted middle and edge symbols.
   * Over-relaxation 1.8 on the channel estimate between EM steps, which gives about twice the convergence speed.
   * **Hyperparameter re-estimation from data.** When all symbols are active (and SNR ≥ 0 dB), a Doppler spectrum is fitted by SBL on a
     121-point Doppler grid. It uses the 2304 data-aided channel snapshots y·x̄*/E|x|², separated per group.
     After that, one EM (posterior-second-moment) update is applied to each group's Rf, and one to each group's Rt.
6. **Stuck-example repair** (SNR ≥ 7.5 dB). At high SNR with fast fading, EM sometimes ends up in a cycle-slipped local
   optimum for a few examples. These examples are outliers in data log-likelihood, found with a robust z-score below -3. Each
   is re-run with 4 alternative schedules: slower rings, left-only, right-only, and higher inflation. The solution with the best MAP
   objective (log-likelihood minus prior energy) is kept.
7. **Safety.** All linear algebra runs in single-threaded torch. Multi-threaded OpenBLAS/torch on the loaded 2-core box was
   10–100× slower and caused the first eval to fail on time. Any exception or non-finite output falls back to LS with
   linear interpolation.

## Why it works
* It uses the right prior, learned from the batch: a sparse delay profile, a Jakes/LOS Doppler model, and a per-group
  model that follows the TDL structure.
* The data REs are 13× more numerous than the pilots. Even at -5 dB, EM with 16-QAM data improves on the pilot-only LMMSE,
  and at ≥15 dB the error is close to what known data would give. That is why the result is past the genie
  pilot-only LMMSE (gap closed about 1.21).
* On my data, feeding the true (genie) statistics into this pipeline no longer helps, so what remains is limited
  by detection, not by the statistics estimates.

## Where it is weak
* **Rich channels at low/mid SNR** (urban_vehicular, ratio about 0.05–0.08 at -5…10 dB). 16-QAM detection at 5–10 dB is
  unreliable. An oracle-data bound is about 40% lower at 10 dB, and I did not find a way to close that gap.
* **Very flat NLOS indoor:** the two-group split costs about 6% compared with a single group.
* **Extreme Doppler near the J0 minimum** (about 450–550 km/h): the lag-9 correlation is ill-conditioned. It is now robust
  (ratio at most 0.14 on B-30ns-500 km/h, down from about 1 before the near-π phase fix), but still the weakest case.
* **Runtime:** about 130 ms on average. Fast-fading, high-SNR, rich cases take about 0.25–0.3 s under load.
* It assumes 16-QAM data, the pilot layout in link_config, and one SNR per batch (it uses mean(no)).

## Sequence of ideas (dev NVE from the official evaluation; "bench" = my own synthetic data)
| # | Change | dev NVE |
|---|---|---|
| – | LMMSE with NNLS-fitted PDP and J0 Doppler (bench only) | bench 0.56 |
| – | M-SBL PDP (bench) | high SNR much better |
| – | Data-aided with hard gated decisions: selection bias, hurt at 10 dB (bench) | – |
| 1 | EM data-aided + numpy linear algebra | FAILURE (2.6 s/call, BLAS thread contention) |
| 2 | Torch single-thread, EM (4 it), SBL Doppler from data, noise-relative rank truncation, fixed J0 inversion sign | 0.0435 |
| 3 | 6 EM iterations, over-relaxation 1.8, time-eigen threshold 1e-5 | 0.0387 |
| 4 | Progressive ring schedule, posterior-variance inflation ×3, data-Doppler gated to ≥7.5 dB, LS fallback | 0.0320 |
| 5 | FN 1.0→0.6, EM updates of Rf and Rt | 0.0291 |
| 6 | Doppler-adaptive ring speed + edge penalty (large gain at 350 km/h on the bench) | 0.0291 |
| 7 | Two-group (dominant path + rest) sum-of-Kronecker prior | 0.0283 |
| 8 | Stuck-example restart repair (fixes bench high-speed outliers; not triggered on dev) | 0.0283 |
| 9 | Data-SBL Doppler down to 0 dB | 0.0281 |
| 10 | Near-π phase read as negative correlation (fixes the 500 km/h failure on the bench) | 0.0281 |
| 11 | Code cleanup (final) | **0.0281** |

Tried and dropped, all on the bench: soft-pilot (non-EM) updates; 4th-moment Doppler estimation; parametric
Jakes+LOS least-squares fits; Toeplitz/nonparametric time covariance (white-component bias); Bayesian averaging over
Doppler; data-aided full-band PDP re-SBL; extrinsic (leave-one-out) demapping; annealed E-step; honest soft-pilot
variance; 3 delay groups; gating the split by power fraction or reliability; Doppler-hypothesis probing; finer
delay grid; more EM or hyperparameter rounds. Each was neutral or worse.
