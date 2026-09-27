# How these estimators were produced (provenance)

All `batched_agent*` and `single_agent*` files were written by AI agents (Claude, model id `claude-opus-5-5`,
run as independent Claude Code "general-purpose" sub-agents inside a Claude Cowork session on 2026-09-26).
Nothing in them was edited by a human, except `ablation_single_agent1_no_energy.py`, which is `single_agent1.py`
with one line changed by the analysis (marked `ABLATION` in the file).

Setup per agent:

- Its own empty workspace containing only `prompt.md` (the task prompt of the track) and `link_config.py`.
- Evaluation only through a logging wrapper: dev split, hard cap of 40 evaluations per agent. Every evaluated
  draft was snapshotted; the `*_evals.jsonl` files list every official evaluation (time, success, dev NVE).
- Instructions: do not read the evaluation harness, baselines, reference results, other agents' workspaces or the
  web; self-generated Sionna data for tuning allowed; leave the best version in `draft.py` and write `NOTES.md`.
  These rules were instructions, not enforced by a sandbox; spot checks of the workspaces found no violations.
- Machine: a 2-core cloud CPU shared by all agents running at the same time. No GPU.
- Batched track: 3 agents in parallel, 1.8-2.5 h each, 11 evaluations used each.
  Single-slot track: 3 agents in parallel, 2.7-3.9 h each, 17 / 15 / 10 evaluations used.
- Token usage reported by the harness: batched 314k / 401k / 375k tokens; single-slot 582k / 502k / 458k tokens.
  (Dollar cost is not reported: the agents ran on a subscription, not a metered API.)

The exact instruction given to each agent is reproduced below (workspace path varies).

---

> You are a wireless signal-processing engineer taking part in a controlled experiment. Your job: design the best
> possible OFDM channel estimator for the task described in `<ws>/prompt.md` [single-slot track: "(the SINGLE-SLOT
> track: your function is scored one slot at a time)"]. Read that file first, and `<ws>/link_config.py`.
>
> Rules of the experiment: Work ONLY inside `<ws>`. Write your estimator to `<ws>/draft.py`. Evaluate ONLY with
> `python evaluate.py <ws>` (dev split, logged, hard budget of 40 evaluations). Do NOT open, read, grep, or import
> anything under the harness / reference directories or other workspaces. Do not look at the evaluation harness,
> baselines, or reference results. Do not search the web. You may use your own knowledge and scratch Python scripts
> inside the workspace (e.g. generate your own synthetic channels with Sionna). The estimator must obey all rules in
> prompt.md. Final results are reported on a held-out split with different random seeds: avoid overfitting to dev
> numbers. Spend real effort: aim for the lowest NVE you can reach.
>
> When done, make sure draft.py contains your BEST version, and write NOTES.md: plain-English explanation of the
> algorithm (what, why it works, weaknesses) and the sequence of ideas tried with their dev NVE.
