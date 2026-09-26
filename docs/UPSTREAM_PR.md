# Contributing the task to NVIDIA's AI Telco Engineer

Both public AITE tasks (`otfs_detector`, `pilotless`) need a GPU and are expensive to iterate on. A CPU-friendly, 4-second-evaluation task is a useful addition for anyone who wants to try the framework without a GPU cluster. That makes it a good first contribution.

## Before opening a pull request

1. Read the repository's contribution rules (`CONTRIBUTING.md` / `SECURITY.md` / license). AITE is under NVIDIA's non-commercial license; check whether NVIDIA accepts external contributions and whether a contributor agreement or DCO sign-off (`git commit -s`) is required. If PRs are not accepted, open an issue proposing the task and link to this repo instead.
2. Run the compatibility test against a fresh checkout:
   ```bash
   git clone https://github.com/NVlabs/the-ai-telco-engineer ../the-ai-telco-engineer
   AITE_REPO=../the-ai-telco-engineer python -m pytest tests/test_aite_compat.py -v
   ```
3. Ideally run the task once with AITE itself (a small `num_workers`, any OpenAI-compatible model) and include the resulting leaderboard screenshot.

## What to put in the PR

- The `tasks/ofdm_channel_estimation/` folder, minus `baselines/` if they prefer (or keep them as references).
- Description: motivation (CPU-only, fast evaluation, stats-agnostic estimation matching the task in the original paper), metric definition, reference numbers table from the README, runtime per evaluation.
- Keep it small and polite; maintainers are busy. Offer to adjust to their conventions.

## Then tell people

- Post on the NVIDIA Aerial forum (6G Developer Program members' forum) with the result figure and a link.
- Mention it in your LinkedIn post about the project.
