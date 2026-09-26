# pocket-telco-engineer

**Can a laptop and a few dollars of LLM calls discover better 6G signal-processing algorithms?**

NVIDIA Research's [*The AI Telco Engineer*](https://github.com/NVlabs/the-ai-telco-engineer) (AITE) showed that swarms of LLM agents can design wireless algorithms that match or beat textbook baselines, using 16 parallel workers, GPUs, Docker and frontier models. `pocket-telco-engineer` (`pte`) asks the budget question: **how much of that capability survives on a MacBook with a small or cheap model, and how does discovery quality scale with money spent?**

It contains:

1. **A new, CPU-friendly benchmark task**: *statistics-agnostic OFDM channel estimation* built on NVIDIA [Sionna 2.x](https://github.com/NVlabs/sionna). Four 3GPP TDL scenarios (flat indoor, dispersive urban, 360 km/h high-speed, rural LOS), 7 SNRs, held-out test split, and a genie LMMSE reference that shows how much headroom exists. One evaluation takes ~4 s on a laptop CPU. The folder is **drop-in compatible with AITE** (`eval_tool.py`, `config.json`, Dockerfile), so the same task runs on NVIDIA's full framework.
2. **A lightweight search loop** re-implementing AITE's idea-driven orchestrator/worker loop in ~600 lines: no Docker, no GPU, no tool calling (so small local models work), any OpenAI-compatible endpoint (Ollama, OpenRouter, OpenAI, vLLM) or Anthropic, **a hard dollar cap**, resumable runs, and full logs.
3. **Experiment tooling**: best-so-far-vs-spend curves, Pareto fronts, held-out re-evaluation, multi-seed study runner, and publication figures.

## Quick start (5 minutes, $0)

```bash
# macOS: brew install python@3.12   (Sionna 2.x needs Python 3.11+)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pte demo                    # full pipeline with a scripted fake LLM: no API key, no cost
open runs/demo*/REPORT.md
python -m pytest -q         # 41 tests, ~2-3 min on a laptop
```

## Evaluate your own estimator

Write a file with `estimate_channel(y, no)` (see [the task prompt](tasks/ofdm_channel_estimation/prompt.md)), then:

```bash
pte eval my_estimator.py              # dev split
pte eval my_estimator.py --split test # held-out split
python scripts/plot_task.py --candidate my_estimator.py --label "mine"   # MSE-vs-SNR figure
```

## Where things stand (reference estimators)

NVE = mean over the 28 (scenario, SNR) points of MSE / MSE of LS + linear interpolation. Lower is better.

| estimator | what it is | dev NVE | test NVE |
|---|---|---|---|
| `ls_nearest` | LS at pilots, nearest-neighbour fill | 2.009 | 2.000 |
| `ls_linear` | LS + linear interpolation (**the reference**) | 1.000 | 1.000 |
| `dft_denoise` | textbook delay-domain denoising | 1.213 | 1.189 |
| demo hybrid | SNR switch between the two above | 0.738 | 0.731 |
| `lmmse_genie` | LMMSE with the *true* channel statistics (not allowed, just a yardstick) | **0.157** | **0.163** |

The gap between ~0.73 and ~0.16 is the prize: an estimator that learns the channel's statistics from the data it receives. Note that the "textbook" DFT method is *worse* than the reference on average because of its error floor on dispersive channels at high SNR ([figure](figures/mse_vs_snr.png)); robustness across scenarios is the whole game.

## Run a real search

Local and free (Ollama):

```bash
ollama pull qwen2.5-coder:7b
pte check-llm configs/ollama-local.yaml
pte run configs/ollama-local.yaml
```

Cheap API (Anthropic Claude Haiku 4.5 at $1/$5 per million tokens as of Sept 2026; always verify prices):

```bash
export ANTHROPIC_API_KEY=...
pte run configs/anthropic.yaml            # hard-capped at $3 by the config
```

Stop with Ctrl+C at any time; rerun the same command to resume. Each run directory contains `leaderboard.jsonl` (every candidate), `candidates/` (every file the model wrote), `transcripts/` (every prompt/reply with token counts and cost), `final.json` (held-out results) and `REPORT.md` + `report.png`.

## The study

[`experiments/budget_scaling.yaml`](experiments/budget_scaling.yaml) runs model x budget arms with 3 independent repeats each, within a $20 total cap:

```bash
python experiments/run_matrix.py experiments/budget_scaling.yaml --dry-run
python experiments/run_matrix.py experiments/budget_scaling.yaml
```

See [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) for the hypotheses, protocol and what counts as a result.

## Running the task on NVIDIA's full framework

```bash
git clone https://github.com/NVlabs/the-ai-telco-engineer
cp -r tasks/ofdm_channel_estimation the-ai-telco-engineer/tasks/
# fill in model + base_url in tasks/ofdm_channel_estimation/config.json, then
python launch.py tasks/ofdm_channel_estimation
```

`tests/test_aite_compat.py` checks this path (set `AITE_REPO=/path/to/the-ai-telco-engineer`).

## Honest limitations

- The subprocess sandbox scrubs API keys from the environment and enforces timeouts, but it is not a security boundary like AITE's containers. LLM-written code runs on your machine.
- The anti-cheating check is a pattern blocklist plus a held-out split; a determined adversarial model could still find loopholes. Inspect the best candidates by hand before claiming results.
- Runtime ("complexity") is wall-clock time and therefore machine-dependent; compare runtimes only within one machine.
- API models are not seeded; "seeds" are independent repeats.

## Credits and licenses

This code: Apache-2.0. It uses [Sionna](https://github.com/NVlabs/sionna) (Apache-2.0) and follows the task format of [The AI Telco Engineer](https://github.com/NVlabs/the-ai-telco-engineer) by Aït Aoudia, Hoydis, Cammerer, Maggi, Marti and Keller (NVIDIA), which is under the NVIDIA non-commercial license. No AITE code is copied here; the compatibility test imports it from your own checkout.
