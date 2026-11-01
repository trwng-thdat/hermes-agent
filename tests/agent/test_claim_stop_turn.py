"""End-to-end coverage for the chat claim gate stop-gate (agent/claim_stop.py)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent

CLAIM = "Đã gửi email cho khách hàng."
HONEST = "Chưa gửi được email vì kênh mail đang lỗi."
FALLBACK_HEAD = "Mình chưa thể xác nhận yêu cầu đã được thực hiện xong."


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
    assert retry[-1]["role"] == "user" and "do not acknowledge, thank or apologise" in retry[-1]["content"]
    assert ("answer_retracted", "answer retracted") in statuses
    assert emitted == []  # never surfaced as an interim message
    assert agent._claim_stop_nudges == 1

    rows = _persisted(db, agent)
    assert rows == [("user", "gửi email cho khách"), ("assistant", HONEST)]
    # Live history handed back for the next turn drops the retracted answer and the nudge too.
    assert [(m["role"], m["content"]) for m in result["messages"]] == rows


def test_budget_exhausted_replaces_claim_with_receipts_summary(agent, db):
    agent.max_iterations = 1
    agent.iteration_budget.max_total = 1
    result, sent, statuses, _ = _run(agent, [CLAIM])

    assert result["final_response"].startswith(FALLBACK_HEAD)
    assert "Chưa có thao tác nào được thực hiện" in result["final_response"]
    assert len(sent) == 1  # no extra model call
    assert ("answer_retracted", "answer retracted") in statuses
    agent._handle_max_iterations.assert_not_called()
    assert _persisted(db, agent) == [("user", "gửi email cho khách"), ("assistant", result["final_response"])]


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


def test_cap_replaces_the_third_answer(agent):
    result, sent, statuses, _ = _run(agent, [CLAIM, CLAIM, CLAIM])

    assert len(sent) == 3
    assert result["final_response"].startswith(FALLBACK_HEAD)
    assert CLAIM not in result["final_response"]
    assert sum(kind == "answer_retracted" for kind, _ in statuses) == 3


def test_nudge_counter_resets_next_turn(agent):
    _run(agent, [CLAIM, HONEST])
    assert agent._claim_stop_nudges == 1
    _run(agent, ["Có 3 kênh Zalo đang hoạt động."], text="có mấy kênh?")
    assert agent._claim_stop_nudges == 0


def test_gate_off_lets_claim_through(agent, monkeypatch):
    monkeypatch.setenv("HERMES_CLAIM_GATE", "0")
    result, sent, _, _ = _run(agent, [CLAIM])
    assert result["final_response"] == CLAIM and len(sent) == 1


def test_nudge_names_failed_writes(monkeypatch):
    from agent.turn_stop_gates import _claim_stop_check

    monkeypatch.setenv("HERMES_CLAIM_GATE", "1")

    def call(cid, name):
        return {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}

    messages = [
        {"role": "user", "content": "tạo nhóm zalo"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("c1", "mcp__aeo__create_zalo_group"), call("c2", "list_department_members"),
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"error": "requires at least two member IDs"}'},
        {"role": "tool", "tool_call_id": "c2", "content": '{"result": "3 members"}'},
    ]
    agent = SimpleNamespace(max_iterations=6, iteration_budget=None, _claim_stop_nudges=0)
    action, nudge = _claim_stop_check(agent, messages, "Đã tạo nhóm Zalo cho bạn.", 0, 1)

    assert action == "nudge"
    assert "FAILED this turn" in nudge and "mcp__aeo__create_zalo_group (c1)" in nudge
    assert "requires at least two member IDs" in nudge
    assert "do not call them again" not in nudge and "list_department_members" not in nudge


def test_honest_answer_after_failure_passes(monkeypatch):
    from agent.turn_stop_gates import _claim_stop_check

    monkeypatch.setenv("HERMES_CLAIM_GATE", "1")
    messages = [
        {"role": "user", "content": "tạo nhóm zalo"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "mcp__aeo__create_zalo_group", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"error": "requires at least two member IDs"}'},
    ]
    agent = SimpleNamespace(max_iterations=6, iteration_budget=None, _claim_stop_nudges=0)
    assert _claim_stop_check(agent, messages, "Chưa tạo được nhóm Zalo vì cần ít nhất hai thành viên.", 0, 1) is None


TAG_TOOL = "mcp__helpdesk__post_customer_by_customer_id_tags"


def _tool_response(call_id, args='{"customer_id": "c_456", "tagID": "vip"}'):
    call = SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=TAG_TOOL, arguments=args))
    msg = SimpleNamespace(content="", tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], model="m", usage=None)


def test_write_receipt_dedup_and_held_stream_through_the_real_loop(agent):
    """The model writes, repeats the same write, then answers: the API is hit once, the repeat
    returns the first receipt, and the answer is withheld from the stream until the gate passes."""
    import json

    executed = []

    def handle_function_call(name, args, *_a, **_kw):
        executed.append((name, args))
        return json.dumps({"result": json.dumps({"status": "OK", "data": {"tagID": "vip"}})})

    queue = iter([_tool_response("c1"), _tool_response("c2"), _response("Đã gắn tag VIP cho khách.")])
    agent._interruptible_api_call = lambda _kw: next(queue)
    agent.valid_tool_names = {TAG_TOOL}
    agent.tools = [{"type": "function", "function": {"name": TAG_TOOL, "parameters": {"type": "object"}}}]
    with (
        patch("model_tools.handle_function_call", side_effect=handle_function_call),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("gắn tag VIP cho khách c_456")

    assert len(executed) == 1
    tool_rows = [m for m in result["messages"] if m.get("role") == "tool"]
    assert '"status": "SUCCEEDED"' in tool_rows[0]["content"]
    assert "Not executed again" in tool_rows[1]["content"] and '"dedup_of": "c1"' in tool_rows[1]["content"]
    assert result["final_response"] == "Đã gắn tag VIP cho khách."
    streamed = []
    agent.stream_delta_callback = lambda text: streamed.append(text)
    agent._fire_stream_delta("sau khi ghi")  # a write ran this turn: held
    assert streamed == []
    agent._current_turn_id = "next-turn"  # no write in this turn yet: streams normally
    agent._fire_stream_delta("lượt mới")
    assert [t.strip() for t in streamed] == ["lượt mới"]
