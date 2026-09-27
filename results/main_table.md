| estimator | origin | batched track (test) | single-slot track (test) | out-of-distribution | runtime [ms] |
|---|---|---|---|---|---|
| ls_linear | reference | 1.0000 | 1.0000 | 1.0000 | 0.2 |
| dft_denoise | textbook | 1.1889 | 1.1889 | 1.2645 | 0.4 |
| demo_hybrid | hand-made | 0.7311 | 0.7311 | 0.7619 | 0.5 |
| batched_agent1 | agent, batched track | 0.0294 | 0.1867 | 0.1300 | 81.2 |
| batched_agent2 | agent, batched track | 0.0276 | 0.1508 | 0.1345 | 112.2 |
| batched_agent3 | agent, batched track | 0.0518 | 0.1153 | 0.1308 | 101.8 |
| single_agent1 | agent, single-slot track | 0.0275 | 0.0275 | 2.3151 | 152.9 |
| single_agent2 | agent, single-slot track | 0.1406 | 0.0369 | 0.0559 | 203.7 |
| single_agent3 | agent, single-slot track | 0.0292 | 0.0292 | 0.0471 | 128.5 |
| single_agent1_ablation_no_energy | ablation (1 line removed) | 0.0285 | 0.0285 | 0.0496 | 143.5 |
| lmmse_genie | not allowed: knows true statistics (pilot-only) | 0.1632 | 0.1632 | 0.1580 | - |
