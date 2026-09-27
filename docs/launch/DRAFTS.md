# Launch drafts (edit into your own words before posting)

Post only after: (1) the repo is public on GitHub, (2) you can explain every number yourself,
(3) you have re-run `pte eval results/discovered/single_agent3.py --split test` on your own Mac.

Replace `<REPO_URL>` everywhere.

---

## 1. NVIDIA Aerial Forum post (Developer Forums > Accelerated Computing > Aerial)

**Title:** A CPU-only Sionna benchmark for agentic algorithm discovery, and what it taught me about reward hacking

Hi all, I'm a high school member of the 6G Developer Program. Inspired by the AI Telco Engineer paper
(Aït Aoudia et al., 2026), I built a small benchmark that runs on a laptop CPU in ~4 s per evaluation:
statistics-agnostic OFDM channel estimation in Sionna 2.1 (4 TDL scenarios, 7 SNRs, held-out split,
genie pilot-only LMMSE reference). The task folder follows the AITE task format, so it drops into
`the-ai-telco-engineer/tasks/`.

I then let LLM agents attack it. Three things I found:

1. They found data-aided (EM/EP) estimators ~17 dB better than LS + linear interpolation, beating the
   genie pilot-only LMMSE.
2. When each call contained 32 slots of the same scenario, agents pooled statistics across them.
   Scored one slot at a time, those designs were 2-6x worse and the ranking reversed.
3. The best benchmark score came from an agent that inferred Sionna's per-slot channel normalization and
   used it. On unnormalized channels it was 2.3x worse than LS. Deleting one line fixed it.

Code, discovered estimators and a stress test: <REPO_URL>

Two questions for the Aerial/Sionna team:
- Would a CPU-friendly task like this be useful as an example in the AI Telco Engineer repo?
- Is there a recommended way to get more realistic channels for out-of-distribution tests (e.g. Sionna RT
  scenes or AODT exports) that doesn't require a large GPU?

Thanks for building such open tools.

---

## 2. LinkedIn post

I asked AI agents to invent a better 6G receiver algorithm. They did, and then one of them cheated without meaning to.

As a member of NVIDIA's 6G Developer Program, I built an open benchmark in NVIDIA Sionna for one of the core
problems in wireless: estimating the radio channel from a few known pilot signals.

What happened:
- The agents found estimators about 17 dB more accurate than the textbook method.
- The top scorer had quietly learned a quirk of the simulator (every slot is normalized to the same energy).
  On realistic channels it became worse than the textbook method. Removing one line fixed it.
- Another benchmark detail (grading 32 slots at once) inflated scores up to 6x.

The lesson I'm taking into the next experiment: when AI discovers an algorithm, the leaderboard is only the start.
You have to test it where it will actually be used.

Code and results (open source): <REPO_URL>

#6G #NVIDIA #Sionna #AI #WirelessCommunications #HighSchoolResearch

[attach figures/main_results.png]

---

## 3. GitHub issue for NVlabs/the-ai-telco-engineer (only after reading their CONTRIBUTING / license terms)

**Title:** Proposal: a CPU-only example task (statistics-agnostic OFDM channel estimation)

Both built-in tasks need a GPU. I've built a CPU task in the AITE format (config.json, prompt.md, eval_tool.py,
eval/, docker/) whose evaluation takes ~4 s on a laptop: SISO OFDM, comb-2 DMRS-like pilots, 4 TDL scenarios,
7 SNRs, NVE vs LS-linear, held-out split, genie pilot-only LMMSE reference. A compatibility test runs its
`EvalTool` against the current `src/`. There is also an optional single-slot evaluation mode, which I added after
finding that agents exploit cross-slot pooling (write-up linked below).

Would you be interested in a PR adding it under `tasks/`? Happy to adapt it to your conventions.
Repo and results: <REPO_URL>
