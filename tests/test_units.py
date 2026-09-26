"""Unit tests for parsing, budget, leaderboard, config, sandbox and LLM clients."""

import json
import os

import httpx
import pytest

from pte.budget import Budget, BudgetExceeded
from pte.config import load_config
from pte.leaderboard import Candidate, Leaderboard
from pte.llm import Anthropic, LLMError, Message, ModelSpec, OpenAICompatible, build_llm
from pte.prompts import extract_code, parse_ideas
from pte.sandbox import run_python, scrubbed_env
from pte.search import _elide_old_drafts
from pte.task import EvalResult

# ----------------------------------------------------------------- parsing


def test_extract_code_prefers_last_block_with_function():
    text = "Plan:\n```python\nimport torch\n```\nFinal:\n```python\ndef estimate_channel(y, no):\n    return y\n```"
    assert extract_code(text).startswith("def estimate_channel")


def test_extract_code_handles_unterminated_fence_and_missing():
    assert "def f" in extract_code("```python\ndef f():\n    pass\n")
    assert extract_code("no code here") is None
    assert "def g" in extract_code("```\ndef g():\n    return 1\n```")


def test_parse_ideas_variants():
    good = '[{"title": "A", "description": "a"}, {"title": "B", "description": "b"}]'
    assert [i["title"] for i in parse_ideas(good, 5)] == ["A", "B"]
    assert len(parse_ideas("Sure!\n```json\n" + good + "\n```", 1)) == 1
    assert parse_ideas('{"ideas": [{"title": "C", "description": "c"}]}', 3)[0]["title"] == "C"
    assert parse_ideas("I cannot do that", 3) == []


def test_eval_result_parse():
    r = EvalResult.parse("warning: blah\nSUCCESS, 0.5, 1.2e-3\nmore")
    assert r.success and r.metric == 0.5 and r.complexity == pytest.approx(1.2e-3)
    assert not EvalResult.parse("FAILURE,\nError: x").success
    assert not EvalResult.parse("Traceback ...").success
    assert not EvalResult.parse("").success

# ------------------------------------------------------------------ budget


def test_budget_caps_spend_and_calls():
    b = Budget(max_usd=0.01, max_calls=100)
    b.check()
    assert b.record(1_000_000, 0, 0.005, 0.0) == pytest.approx(0.005)
    b.check()
    b.record(0, 1_000_000, 0.0, 0.006)
    with pytest.raises(BudgetExceeded):
        b.check()
    b2 = Budget(max_usd=1.0, max_calls=1)
    b2.record(1, 1, 0, 0)
    with pytest.raises(BudgetExceeded):
        b2.check()
    b3 = Budget()
    b3.restore(b.snapshot())
    assert b3.spent_usd == pytest.approx(0.011) and b3.calls == 2

# ------------------------------------------------------------- leaderboard


def _cand(i, metric, cx, ok=True, idea="x", spent=0.0):
    return Candidate(id=f"c{i}", generation=0, worker=0, step=i, idea=idea, success=ok,
                     metric=metric if ok else None, complexity=cx if ok else None, code_path="",
                     spent_usd=spent, llm_calls=i, wall_s=float(i), error="" if ok else "boom")


def test_leaderboard_best_pareto_curve_and_persistence(tmp_path):
    lb = Leaderboard(tmp_path / "lb.jsonl")
    for c in [_cand(1, 1.0, 1e-3, spent=0.1), _cand(2, 0.8, 5e-3, spent=0.2), _cand(3, 0.9, 5e-4, spent=0.3),
              _cand(4, 0.95, 6e-3, spent=0.4), _cand(5, None, None, ok=False, idea="bad", spent=0.5)]:
        lb.add(c)
    assert lb.best_one().id == "c2"
    assert [c.id for c in lb.pareto_front()] == ["c3", "c2"]          # c1 and c4 are dominated
    assert [p[3] for p in lb.best_so_far_curve()] == [1.0, 0.8, 0.8, 0.8, 0.8]
    assert "FAILED" in lb.summary() and "0.8000" in lb.summary()
    again = Leaderboard(tmp_path / "lb.jsonl")
    assert len(again.items) == 5 and again.best_one().id == "c2"


def test_leaderboard_higher_is_better(tmp_path):
    lb = Leaderboard(tmp_path / "lb.jsonl", higher_is_better=True)
    lb.add(_cand(1, 1.0, 1e-3))
    lb.add(_cand(2, 2.0, 1e-3))
    assert lb.best_one().id == "c2"
    assert [p[3] for p in lb.best_so_far_curve()] == [1.0, 2.0]

# ------------------------------------------------------------------ config


def test_config_env_interpolation_and_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("PTE_TEST_MODEL", "my-model")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "task: ../task\n"
        "models:\n  worker:\n    provider: openai\n    model: ${PTE_TEST_MODEL}\n"
        "    base_url: ${PTE_UNSET_URL:-http://localhost:11434/v1}\n    price_in_per_mtok: ${PTE_UNSET_P:-0.5}\n"
        "search:\n  generations: 2\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.worker.model == "my-model" and cfg.orchestrator.model == "my-model"
    assert cfg.worker.base_url == "http://localhost:11434/v1"
    assert cfg.worker.price_in_per_mtok == 0.5
    assert cfg.search.generations == 2 and cfg.search.steps_per_worker == 4
    assert cfg.task == str((tmp_path / "../task").resolve())
    cfg_file.write_text("task: t\nmodels:\n  worker: {provider: openai}\nsearch:\n  typo_key: 1\n")
    with pytest.raises(ValueError, match="typo_key"):
        load_config(cfg_file)


def test_example_configs_load():
    from conftest import REPO

    for f in sorted((REPO / "configs").glob("*.yaml")):
        cfg = load_config(f)
        assert os.path.isdir(cfg.task), f
        assert cfg.worker.provider in {"openai", "anthropic"}

# ----------------------------------------------------------------- sandbox


def test_sandbox_scrubs_secrets_and_times_out(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "sk-secret")
    monkeypatch.setenv("HARMLESS", "ok")
    env = scrubbed_env()
    assert "MY_API_KEY" not in env and env["HARMLESS"] == "ok"
    (tmp_path / "leak.py").write_text("import os; print(os.environ.get('MY_API_KEY'))")
    assert run_python(["leak.py"], tmp_path, 30).output.strip() == "None"
    (tmp_path / "spin.py").write_text("while True: pass")
    res = run_python(["spin.py"], tmp_path, 1.0)
    assert res.timed_out


def test_elide_old_drafts():
    msgs = [Message("user", "task")]
    for i in range(4):
        msgs += [Message("assistant", f"v{i}\n```python\ncode{i}\n```"), Message("user", f"fb{i}")]
    out = _elide_old_drafts(msgs, keep_last=2)
    texts = [m.content for m in out if m.role == "assistant"]
    assert "code0" not in texts[0] and "elided" in texts[0] and "v0" in texts[0]
    assert "code2" in texts[2] and "code3" in texts[3]

# -------------------------------------------------------------- LLM clients


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_compatible_client_and_retry(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "abc")
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, text="slow down")
        body = json.loads(request.content)
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        assert request.headers["authorization"] == "Bearer abc"
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}],
                                         "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

    spec = ModelSpec(provider="openai", model="m", base_url="http://x/v1", api_key_env="TEST_KEY")
    llm = OpenAICompatible(spec, client=_mock_client(handler), sleep=lambda s: None)
    r = llm.complete("sys", [Message("user", "hello")])
    assert (r.text, r.input_tokens, r.output_tokens) == ("hi", 7, 3)
    assert len(calls) == 2 and str(calls[1].url) == "http://x/v1/chat/completions"


def test_openai_client_fatal_error_and_missing_key(monkeypatch):
    spec = ModelSpec(provider="openai", model="m", base_url="http://x/v1")
    llm = OpenAICompatible(spec, client=_mock_client(lambda r: httpx.Response(400, text="bad")),
                           sleep=lambda s: None)
    with pytest.raises(LLMError, match="400"):
        llm.complete("s", [Message("user", "u")])
    monkeypatch.delenv("NOPE_KEY", raising=False)
    spec2 = ModelSpec(provider="openai", model="m", base_url="http://x/v1", api_key_env="NOPE_KEY")
    with pytest.raises(LLMError, match="NOPE_KEY"):
        OpenAICompatible(spec2, client=_mock_client(lambda r: httpx.Response(200))).complete("s", [])


def test_anthropic_client(monkeypatch):
    monkeypatch.setenv("ANT_KEY", "k")

    def handler(request):
        body = json.loads(request.content)
        assert body["system"] == "sys" and request.headers["x-api-key"] == "k"
        assert request.headers["anthropic-version"] == "2023-06-01"
        return httpx.Response(200, json={"content": [{"type": "text", "text": "yo"}],
                                         "usage": {"input_tokens": 5, "output_tokens": 2}})

    spec = ModelSpec(provider="anthropic", model="claude-x", api_key_env="ANT_KEY")
    r = Anthropic(spec, client=_mock_client(handler)).complete("sys", [Message("user", "u")])
    assert (r.text, r.input_tokens, r.output_tokens) == ("yo", 5, 2)


def test_build_llm_validation():
    with pytest.raises(ValueError):
        build_llm(ModelSpec(provider="openai", model="m"))
    with pytest.raises(ValueError):
        build_llm(ModelSpec(provider="nope"))
