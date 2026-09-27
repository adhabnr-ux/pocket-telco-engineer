"""The evaluation harness is the ground truth of every experiment: test it hard."""

import importlib.util
import json

import numpy as np
import pytest
import torch

from conftest import BASELINES, EVAL_DIR


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem + "_t", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.estimate_channel


def test_generation_is_deterministic_and_seed_dependent(ev):
    sc = ev.SCENARIOS[2]
    a = ev.generate_batch(sc, 10.0, 4, seed=11)
    b = ev.generate_batch(sc, 10.0, 4, seed=11)
    c = ev.generate_batch(sc, 10.0, 4, seed=12)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.equal(a[2], c[2])


def test_shapes_power_and_pilots(ev):
    from link_config import PILOT_MASK, PILOT_SYMBOLS

    y, no, h = ev.generate_batch(ev.SCENARIOS[0], 5.0, 16, seed=3, noiseless=True)
    assert y.shape == h.shape == (16, 14, 72) and no.shape == (16,)
    assert abs(float((h.abs() ** 2).mean()) - 1.0) < 1e-3          # normalized channel
    m, p = torch.as_tensor(PILOT_MASK), torch.as_tensor(PILOT_SYMBOLS)
    assert torch.allclose(y[:, m], h[:, m] * p[m], atol=1e-5)       # y = h * x on pilots (noiseless)


def test_noise_level_matches_snr(ev):
    y, no, h = ev.generate_batch(ev.SCENARIOS[1], 10.0, 64, seed=5)
    y0, _, h0 = ev.generate_batch(ev.SCENARIOS[1], 10.0, 64, seed=5, noiseless=True)
    assert torch.equal(h, h0)
    from link_config import PILOT_MASK, PILOT_SYMBOLS

    m, p = torch.as_tensor(PILOT_MASK), torch.as_tensor(PILOT_SYMBOLS)
    noise = y[:, m] - h[:, m] * p[m]
    assert float((noise.abs() ** 2).mean()) == pytest.approx(0.1, rel=0.1)
    assert float(no[0]) == pytest.approx(0.1)


def test_splits_do_not_overlap(ev):
    seeds = {s: {ev._point_seed(s, i, j) for i in range(len(ev.SCENARIOS)) for j in range(len(ev.SNRS_DB))}
             for s in ev.SPLITS}
    assert not (seeds["dev"] & seeds["test"])


def test_reference_nves_are_consistent(ev):
    base = json.loads((EVAL_DIR / "baseline_mse.json").read_text())
    for split in ("dev", "test"):
        ref = base[split]["ls_linear"]
        assert set(ref) == {ev.point_key(sc, snr) for sc in ev.SCENARIOS for snr in ev.SNRS_DB}
        nve = {n: np.mean([v[k] / ref[k] for k in ref]) for n, v in base[split].items()}
        assert nve["ls_linear"] == pytest.approx(1.0)
        assert nve["ls_nearest"] > 1.5
        assert nve["lmmse_genie"] < 0.3                       # plenty of headroom for discovery
        # the genie is (near-)optimal: it beats LS-linear at every single point
        assert all(base[split]["lmmse_genie"][k] < ref[k] for k in ref)


def test_baseline_file_matches_current_harness(ev):
    """If the harness changes, baseline_mse.json must be rebuilt (python build_baseline.py)."""
    base = json.loads((EVAL_DIR / "baseline_mse.json").read_text())
    fn = _load(BASELINES / "ls_linear.py")
    sc, snr = ev.SCENARIOS[1], ev.SNRS_DB[3]
    y, no, h = ev.generate_batch(sc, snr, ev.SPLITS["dev"]["batch_size"], ev._point_seed("dev", 1, 3))
    assert ev.mse(fn(y, no), h) == pytest.approx(base["dev"]["ls_linear"][ev.point_key(sc, snr)], rel=1e-5)


def test_torch_ls_linear_agrees_with_sionna_inside_pilot_span(ev):
    """Cross-check our reference against Sionna's own LS + linear interpolation."""
    from sionna.phy.ofdm import LSChannelEstimator, PilotPattern, ResourceGrid
    from link_config import PILOT_MASK, PILOT_SYMBOLS

    mask = torch.as_tensor(PILOT_MASK).reshape(1, 1, 14, 72)
    pilots = torch.as_tensor(PILOT_SYMBOLS)[torch.as_tensor(PILOT_MASK)].reshape(1, 1, -1)
    rg = ResourceGrid(num_ofdm_symbols=14, fft_size=72, subcarrier_spacing=30e3,
                      pilot_pattern=PilotPattern(mask, pilots))
    y, no, h = ev.generate_batch(ev.SCENARIOS[1], 15.0, 8, seed=77)
    h_sionna, _ = LSChannelEstimator(rg, interpolation_type="lin")(y.reshape(8, 1, 1, 14, 72), 0.0316)
    h_sionna = h_sionna.reshape(8, 14, 72)
    ours = _load(BASELINES / "ls_linear.py")(y, no)
    inner = (slice(None), slice(2, 12), slice(0, 71))  # between pilot symbols / subcarriers (no extrapolation)
    assert torch.allclose(ours[inner], h_sionna[inner], atol=1e-4)


@pytest.mark.parametrize("name,lo,hi", [("ls_linear", 0.999, 1.001), ("ls_nearest", 1.5, 3.0),
                                        ("dft_denoise", 1.0, 1.5)])
def test_cli_scores_baselines(task, name, lo, hi):
    res = task.evaluate_file(BASELINES / f"{name}.py")
    assert res.success, res.output
    assert lo <= res.metric <= hi
    assert 0 < res.complexity < 0.5
    assert res.details["split"] == "dev" and len(res.details["points"]) == 28


BAD_CANDIDATES = {
    "wrong_shape": ("def estimate_channel(y, no):\n    return y[:, :2]\n", "shape"),
    "real_valued": ("def estimate_channel(y, no):\n    return y.real\n", "complex"),
    "nan": ("import torch\ndef estimate_channel(y, no):\n    return y * float('nan')\n", "NaN"),
    "missing_fn": ("def estimate(y, no):\n    return y\n", "must define"),
    "raises": ("def estimate_channel(y, no):\n    raise ValueError('boom')\n", "boom"),
    "syntax": ("def estimate_channel(y, no)\n    return y\n", "SyntaxError"),
    "introspection": ("import inspect\ndef estimate_channel(y, no):\n    return y\n", "forbidden"),
    "reads_reference": ("X = open('baseline_mse.json').read()\ndef estimate_channel(y, no):\n    return y\n",
                        "forbidden"),
    "too_slow": ("import time\ndef estimate_channel(y, no):\n    time.sleep(0.6)\n    return y\n", "too slow"),
    "not_a_tensor": ("def estimate_channel(y, no):\n    return [1, 2]\n", "tensor or ndarray"),
}


@pytest.mark.parametrize("case", sorted(BAD_CANDIDATES))
def test_bad_candidates_fail_cleanly(task, case):
    code, needle = BAD_CANDIDATES[case]
    res = task.evaluate_code(code, timeout_s=120)
    assert not res.success
    assert res.output.startswith("FAILURE,")
    assert needle in res.output, res.output


def test_numpy_return_is_accepted(task):
    code = ("import numpy as np\nfrom link_config import *\n"
            "def estimate_channel(y, no):\n    return np.ones((y.shape[0], 14, 72), dtype=np.complex64)\n")
    res = task.evaluate_code(code)
    assert res.success, res.output


def test_held_out_split(task):
    res = task.evaluate_file(BASELINES / "ls_linear.py", split="test")
    assert res.success and res.metric == pytest.approx(1.0, abs=1e-6)


def test_single_slot_track_shares_harness_and_scores_reference():
    from conftest import REPO
    from pte.task import Task

    single = Task(REPO / "tasks" / "ofdm_channel_estimation_single_slot")
    assert single.eval_dir == (REPO / "tasks" / "ofdm_channel_estimation" / "eval").resolve()
    assert single.eval_args == ["--chunk", "1"]
    assert "ONE slot at a time" in single.prompt
    res = single.evaluate_file(BASELINES / "ls_linear.py")
    assert res.success and res.metric == pytest.approx(1.0, abs=1e-6)
    assert res.details["chunk"] == 1


def test_chunking_exposes_batch_pooling(task):
    """An estimator that pools statistics across the batch scores differently when called one slot at a time."""
    code = ("import torch\nfrom link_config import *\n"
            "def estimate_channel(y, no):\n"
            "    # cheat-ish: average the pilot-LS estimate over the whole batch\n"
            "    m = torch.as_tensor(PILOT_MASK); p = torch.as_tensor(PILOT_SYMBOLS)\n"
            "    ls = (y[:, m] / p[m]).mean(0, keepdim=True)\n"
            "    g = ls.mean().expand(y.shape[0], 14, 72).clone()\n"
            "    return g.to(torch.complex64)\n")
    import pathlib
    import tempfile

    batched = task.evaluate_code(code)
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "c.py"
        f.write_text(code)
        single = task.evaluate_file(f, extra_args=["--chunk", "1", "--no-timing"])
    assert batched.success and single.success
    assert batched.metric != pytest.approx(single.metric, rel=1e-3)
