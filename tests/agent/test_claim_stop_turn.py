"""End-to-end coverage for the chat claim gate stop-gate (agent/claim_stop.py)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent

CLAIM = "Đã gửi email cho khách hàng."
HONEST = "Chưa gửi được email vì kênh mail đang lỗi."


def _response(content):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


@pytest.fixture
def db(tmp_path):
    store = SessionDB(db_path=tmp_path / "state.db")
    yield store
    store.close()


@pytest.fixture
def agent(tmp_path, monkeypatch, db):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    monkeypatch.setenv("HERMES_CLAIM_GATE", "1")
    monkeypatch.setenv("HERMES_CLAIM_JUDGE", "0")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            session_id="claim-gate-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=6,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=db,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    instance._handle_max_iterations = MagicMock(return_value="replacement summary")
    return instance


def _run(agent, answers, text="gửi email cho khách"):
    queue = iter(answers)
    sent = []

    def model_call(api_kwargs):
        sent.append(api_kwargs.get("messages"))
        return _response(next(queue))

    agent._interruptible_api_call = model_call
    statuses = []
    agent.status_callback = lambda kind, msg=None, **_kw: statuses.append((kind, msg))
    emitted = []
    agent.interim_assistant_callback = lambda t, **_kw: emitted.append(t)
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation(text)
    return result, sent, statuses, emitted


def _persisted(db, agent):
    return [(m["role"], m["content"]) for m in db.get_messages_as_conversation(agent.session_id)]


def test_unbacked_claim_is_retracted_and_corrected(agent, db):
    result, sent, statuses, emitted = _run(agent, [CLAIM, HONEST])

    assert result["final_response"] == HONEST
    assert len(sent) == 2
    # The retry saw the rejected answer and the nudge.
    retry = sent[1]
    assert retry[-2]["role"] == "assistant" and CLAIM in str(retry[-2]["content"])
    assert retry[-1]["role"] == "user" and "Call the appropriate tool now" in retry[-1]["content"]
    assert ("answer_retracted", "answer retracted") in statuses
    assert emitted == []  # never surfaced as an interim message
    assert agent._claim_stop_nudges == 1

    rows = _persisted(db, agent)
    assert rows == [("user", "gửi email cho khách"), ("assistant", HONEST)]
    # Live history handed back for the next turn drops the retracted answer and the nudge too.
    assert [(m["role"], m["content"]) for m in result["messages"]] == rows


def test_gate_skips_when_budget_cannot_fund_retry(agent, db):
    agent.max_iterations = 1
    agent.iteration_budget.max_total = 1
    result, sent, statuses, _ = _run(agent, [CLAIM])

    assert result["final_response"] == CLAIM
    assert len(sent) == 1
    assert not any(kind == "answer_retracted" for kind, _ in statuses)
    agent._handle_max_iterations.assert_not_called()
    assert _persisted(db, agent) == [("user", "gửi email cho khách"), ("assistant", CLAIM)]


def test_retracted_answer_is_never_the_budget_fallback(agent, db):
    # Gate allows the nudge (3 calls budget), then the retry burns the rest on tool calls.
    agent.max_iterations = 3
    agent.iteration_budget.max_total = 3
    tool_call = SimpleNamespace(
        id="c1", type="function", function=SimpleNamespace(name="nope", arguments="{}")
    )
    calls = {"n": 0}

    def model_call(_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _response(CLAIM)
        msg = SimpleNamespace(content="", tool_calls=[tool_call])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], model="m", usage=None
        )

    agent._interruptible_api_call = model_call
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation("gửi email cho khách")
    assert result["final_response"] != CLAIM


def test_cap_lets_the_third_answer_out(agent):
    result, sent, statuses, _ = _run(agent, [CLAIM, CLAIM, CLAIM])

    assert len(sent) == 3
    assert result["final_response"] == CLAIM
    assert sum(kind == "answer_retracted" for kind, _ in statuses) == 2


def test_nudge_counter_resets_next_turn(agent):
    _run(agent, [CLAIM, HONEST])
    assert agent._claim_stop_nudges == 1
    _run(agent, ["Có 3 kênh Zalo đang hoạt động."], text="có mấy kênh?")
    assert agent._claim_stop_nudges == 0


def test_gate_off_lets_claim_through(agent, monkeypatch):
    monkeypatch.setenv("HERMES_CLAIM_GATE", "0")
    result, sent, _, _ = _run(agent, [CLAIM])
    assert result["final_response"] == CLAIM and len(sent) == 1


LONG_FACT = "Hiện có 6 ticket đang mở trong hệ thống hỗ trợ, trong đó 2 ticket ưu tiên cao cần xử lý ngay."
LONG_FIXED = "Hiện có 5 ticket đang mở trong hệ thống hỗ trợ, trong đó 2 ticket ưu tiên cao cần xử lý ngay."


def test_judge_contradiction_then_ok(agent, monkeypatch):
    monkeypatch.setenv("HERMES_CLAIM_JUDGE", "1")
    verdicts = iter([
        {"verdict": "CONTRADICTED", "issues": [{"claim": "6 ticket", "why": "tool returned 5"}]},
        {"verdict": "OK", "issues": []},
    ])
    with patch("agent.claim_stop.judge_grounding", side_effect=lambda *_a, **_kw: next(verdicts)) as judge:
        result, sent, _, _ = _run(agent, [LONG_FACT, LONG_FIXED], text="có bao nhiêu ticket mở?")

    assert judge.call_count == 2
    assert result["final_response"] == LONG_FIXED
    assert "6 ticket" in sent[1][-1]["content"] and "tool returned 5" in sent[1][-1]["content"]


def test_judge_ok_overrides_rule_hit(agent, monkeypatch):
    monkeypatch.setenv("HERMES_CLAIM_JUDGE", "1")
    with patch("agent.claim_stop.judge_grounding", return_value={"verdict": "OK", "issues": []}) as judge:
        result, sent, _, _ = _run(agent, [CLAIM])
    assert judge.call_count == 1
    assert judge.call_args.kwargs["rule_hint"].startswith("Completion claim")
    assert result["final_response"] == CLAIM and len(sent) == 1


def test_judge_error_fails_open_but_rule_still_applies(agent, monkeypatch):
    monkeypatch.setenv("HERMES_CLAIM_JUDGE", "1")
    with patch("agent.oneshot.run_oneshot", side_effect=RuntimeError("no provider")):
        # No completion claim: judge failure means no nudge.
        result, sent, _, _ = _run(agent, [LONG_FACT], text="có bao nhiêu ticket mở?")
        assert result["final_response"] == LONG_FACT and len(sent) == 1
        # Completion claim: the rule nudges on its own.
        result, sent, _, _ = _run(agent, [CLAIM, HONEST])
        assert result["final_response"] == HONEST and len(sent) == 2


def test_interrupt_during_judge_skips_nudge(agent, monkeypatch):
    monkeypatch.setenv("HERMES_CLAIM_JUDGE", "1")

    def judge(*_a, **_kw):
        agent._interrupt_requested = True
        return {"verdict": "CONTRADICTED", "issues": [{"claim": "x", "why": "y"}]}

    with patch("agent.claim_stop.judge_grounding", side_effect=judge):
        _result, sent, statuses, _ = _run(agent, [LONG_FACT], text="có bao nhiêu ticket mở?")
    assert len(sent) == 1
    assert not any(kind == "answer_retracted" for kind, _ in statuses)
