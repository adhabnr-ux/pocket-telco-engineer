# Single-slot track

Same links, scenarios, SNRs, splits, metric and reference numbers as `../ofdm_channel_estimation`, but the
estimator is scored one slot at a time (`eval.py --chunk 1`). This removes a structural shortcut of the
batched task, where all 32 slots of a call share one scenario and SNR and an estimator can pool channel
statistics across them. Real receivers see one slot of one user at a time, so this is the realistic track.

pte-only variant (it reuses the harness of the batched task via `pte_eval_dir`); to run it under NVIDIA's AITE,
copy `../ofdm_channel_estimation` and change the eval command in its `eval_tool.py` to `eval.py draft.py --chunk 1`.
