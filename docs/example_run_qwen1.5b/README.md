# Example: a real run with a tiny open model

Qwen2.5-Coder-1.5B-Instruct served on 2 CPU cores (no GPU), 1 generation, 2 workers x 2 steps, 12.6 minutes, $0.

Result: 0 of 4 candidates evaluated. The failures are instructive:

- `ModuleNotFoundError: sklearn`: the model assumed scikit-learn was installed.
- `sionna.ofdm`: the model knows the old Sionna 0.x import paths, not Sionna 2.x (`sionna.phy.ofdm`).
- `sys.modules`: blocked by the anti-cheating rules.

After this run, the task prompt was updated to list the available packages and the Sionna 2.x import path. How much small models improve from that one change is itself worth measuring (7B+ models via Ollama are the realistic local baseline).
