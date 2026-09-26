# Research plan: agentic algorithm discovery on a student budget

## Question

NVIDIA's AI Telco Engineer (Aït Aoudia et al., 2026; arXiv:2604.19803 and follow-up arXiv:2607.17762) used frontier models (GPT-5.x), 16-20 parallel workers, GPUs and up to 100 generations. Nobody has measured the **cost curve**: how much estimator quality you get per dollar, and whether small open models running on a laptop can discover anything useful at all.

**Primary question.** For statistics-agnostic OFDM channel estimation, how does the best discovered estimator's held-out NVE depend on (a) the model and (b) the LLM budget?

## Hypotheses (write them down *before* running, so you cannot move the goalposts)

- **H1.** A cheap API model (Claude Haiku 4.5) beats the reference estimator (test NVE < 1.0) in every repeat with a $1 budget.
- **H2.** Quality improves roughly logarithmically with budget: quadrupling spend ($1 → $4) closes a further, smaller fraction of the gap to the genie LMMSE.
- **H3.** A 7B local model finds valid estimators but rarely beats the simple SNR-switched hybrid (test NVE 0.731).
- **H4.** Showing workers the current best solution (exploitation) helps at small budgets.

## Protocol

1. Freeze the task: `tasks/ofdm_channel_estimation` at a tagged git commit. Never change the evaluation after seeing results (if you must, re-run everything and say so).
2. Run `experiments/budget_scaling.yaml`: every arm x 3 repeats. Record the git commit, date, model IDs and prices in your lab notebook.
3. Report **held-out test NVE** of the best dev candidate of each run (from `final.json`), as mean ± spread over repeats. Dev numbers are for the search, not for claims.
4. Main figure: best-so-far NVE vs cumulative spend, all runs overlaid (`pte compare`).
5. Secondary: success rate (fraction of candidates that evaluate), Pareto front (NVE vs runtime), fraction of the LS → genie-LMMSE gap closed.
6. Read the top candidates. Explain in plain words *what algorithm* the model found and *why* it works (e.g. "estimates the frequency covariance from the batch, then Wiener-filters"). This explanation is what judges and reviewers care about most.
7. Sanity checks for every claimed winner: runs on the test split, contains no blocklisted constructs, still works with a batch size of 1, and beats the reference in every scenario or you say where it loses.

## Controls and threats to validity

- **Luck:** 3+ repeats per arm; report the spread.
- **Overfitting to the dev draws:** the test split uses different seeds and 4x more samples.
- **Cheating / reward hacking:** blocklist + held-out split + manual code reading of every reported winner.
- **Price drift:** record the per-token prices you used; report tokens as well as dollars.
- **Machine dependence:** runtime is compared only within one machine.

## Possible extensions (each is a paper-sized idea)

- More tasks: link adaptation (MCS selection), MIMO detection, or a pilot-pattern design task where the agent chooses *where* the pilots go.
- Scaling "up": run the same task on NVIDIA's full AITE with bigger budgets on rented GPUs and join the curves.
- Distillation: can a strong model's discovered algorithm be explained well enough that a small model re-derives it?
- Put the best estimator into a real receiver chain (LDPC-coded BLER instead of MSE) with Sionna, or onto the Aerial Testbed as a dApp.

## Deliverables and venues

- A short paper (arXiv; IEEE student workshops; the NVIDIA Aerial forum).
- A science-fair project (Systems Software or Engineering Technology categories): the question, the cost curve, and an explained discovered algorithm.
- An upstream pull request of the task to NVlabs/the-ai-telco-engineer (see docs/UPSTREAM_PR.md).
