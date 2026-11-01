"""Unit tests for the execution ledger (agent/execution_ledger.py): three-state receipts decided
in code, postcondition checks, and duplicate-write prevention."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent import execution_ledger as ledger_mod
from agent.display import _detect_tool_failure
from agent.execution_ledger import (
    FAILED, RECEIPT_KEY, SUCCEEDED, UNKNOWN, after_execute, before_execute, classify, ledger_for,
    postcondition_mismatch,
)

TAG = "mcp__helpdesk__post_customer_by_customer_id_tags"
READ = "mcp__helpdesk__get_customer_by_customer_id"
ARGS = {"customer_id": "c_456", "tagID": "vip"}


def _mcp(body) -> str:
    """An MCP text result as the runtime renders it: the API body JSON-encoded under "result"."""
    return json.dumps({"result": json.dumps(body, ensure_ascii=False)}, ensure_ascii=False)


def _agent(turn="t1"):
    return SimpleNamespace(_current_turn_id=turn)


# --- classify ------------------------------------------------------------------------------

@pytest.mark.parametrize("result,status", [
    (_mcp({"status": "OK", "data": {"tagID": "vip"}, "message": "Tạo thành công"}), SUCCEEDED),
    (_mcp({"success": False, "message": "tag locked"}), FAILED),
    (_mcp({"statusCode": 400, "error": "Bad Request"}), FAILED),
    (_mcp({"status": "FAILED", "errorMessage": "x"}), FAILED),
    (_mcp({"status": "pending", "jobId": "j1"}), UNKNOWN),
    (_mcp({"statusCode": 202}), UNKNOWN),
    ('{"error": "400: Customer Tag đã tồn tại"}', FAILED),
    ('{"error": "Internal error while reading data"}', UNKNOWN),
    ('{"error": "The query took too long; narrow the filters or the date range"}', UNKNOWN),
    ('{"error": "MCP call timed out after 30.0s", "outcome_uncertain": true}', UNKNOWN),
    ('{"error": "MCP call failed: ValueError: bad"}', FAILED),
    ("Error executing tool 'x': boom", FAILED),
    ("sent", SUCCEEDED),
])
def test_classify_three_states(result, status):
    assert classify(TAG, result, ARGS)[0] == status


@pytest.mark.parametrize("name,write", [
    ("mcp__helpdesk__post_customer_by_customer_id_tags", True),
    ("mcp__helpdesk_all__post_customer_by_customer_id_tags", True),
    ("mcp__helpdesk_all__post_marketing_templates_generate_templa_3f9a1c", True),
    ("mcp__helpdesk_all__get_customer_by_customer_id", False),
    ("mcp__jarvis-helpdesk__delete_tag_by_id", True),
])
def test_helpdesk_server_variants(name, write):
    assert ledger_mod.is_write_tool(name) is write


def test_http_200_with_failure_body_is_a_failure_for_display_too():
    assert _detect_tool_failure(TAG, _mcp({"success": False, "message": "tag locked"}))[0] is True
    assert _detect_tool_failure(TAG, _mcp({"status": "OK", "data": {}}))[0] is False
    assert _detect_tool_failure(TAG, _mcp({"error": None, "data": []}))[0] is False


def test_scenario5_postcondition_not_met():
    body = {"status": "OK", "data": {"tagID": "normal"}}
    status, verified, evidence, error = classify(TAG, _mcp(body), ARGS)
    assert (status, verified, evidence) == (FAILED, True, "postcondition not met")
    assert "tagID" in error
    assert postcondition_mismatch({"tags": ["VIP", "New"]}, {"data": {"tags": ["vip"]}}).startswith("tags")
    assert postcondition_mismatch({"tags": ["VIP"]}, {"data": {"tags": ["VIP", "Old"]}}) is None
    # Long text may come back reformatted: never compared.
    assert postcondition_mismatch({"content": "x" * 500}, {"content": "y" * 500}) is None
    # Keys the response does not echo cannot be checked.
    assert postcondition_mismatch({"customer_id": "c_456"}, {"data": {"customerID": "c_456"}}) is None


# --- ledger hooks --------------------------------------------------------------------------

def test_scenario2_receipt_is_merged_into_the_result_without_a_read_back():
    agent = _agent()
    result = after_execute(agent, TAG, ARGS, "call_1", _mcp({"status": "OK", "data": {"tagID": "vip"}}))
    receipt = json.loads(result)[RECEIPT_KEY]
    assert receipt == {
        "operation_id": "call_1", "tool": TAG, "status": SUCCEEDED, "verified": True,
        "evidence": "authoritative API response",
    }
    assert [r.status for r in agent._execution_ledger.receipts] == [SUCCEEDED]


def test_read_tools_get_no_receipt():
    agent = _agent()
    body = _mcp({"data": {"tags": ["vip"]}})
    assert after_execute(agent, READ, {"customer_id": "c_456"}, "r1", body) == body
    assert ledger_for(agent).receipts == []


def test_scenario7_succeeded_write_is_not_executed_again():
    agent = _agent()
    after_execute(agent, TAG, ARGS, "call_1", _mcp({"status": "OK", "data": {"tagID": "vip"}}))
    dup = before_execute(agent, TAG, dict(reversed(list(ARGS.items()))), "call_2")  # same args, other order
    data = json.loads(dup)
    assert "Not executed again" in data["result"]
    assert data[RECEIPT_KEY]["status"] == SUCCEEDED and data[RECEIPT_KEY]["dedup_of"] == "call_1"
    assert _detect_tool_failure(TAG, dup)[0] is False
    # Different arguments are a different operation.
    assert before_execute(agent, TAG, {**ARGS, "tagID": "new"}, "call_3") is None


def test_scenario4_unknown_write_needs_a_state_check_before_retry():
    agent = _agent()
    timeout = '{"error": "MCP call timed out after 30.0s", "outcome_uncertain": true}'
    after_execute(agent, TAG, ARGS, "call_1", timeout)
    blocked = json.loads(before_execute(agent, TAG, ARGS, "call_2"))
    assert blocked["outcome_uncertain"] is True and "Check the current state" in blocked["error"]
    # A read checked the state: the retry is allowed now.
    assert before_execute(agent, READ, {"customer_id": "c_456"}, "r1") is None
    assert before_execute(agent, TAG, ARGS, "call_3") is None


def test_already_exists_after_unknown_means_the_first_call_landed():
    agent = _agent()
    after_execute(agent, TAG, ARGS, "call_1", '{"error": "Internal error while reading data"}')
    before_execute(agent, READ, {}, "r1")
    result = after_execute(agent, TAG, ARGS, "call_2", '{"error": "400: Customer Tag đã tồn tại"}')
    receipt = json.loads(result)[RECEIPT_KEY]
    assert receipt["status"] == SUCCEEDED and receipt["evidence"] == "already applied by call_1"
    # Without an earlier UNKNOWN the same refusal stays a failure.
    other = _agent("t2")
    result = after_execute(other, TAG, ARGS, "call_9", '{"error": "400: Customer Tag đã tồn tại"}')
    assert json.loads(result)[RECEIPT_KEY]["status"] == FAILED


def test_scenario6_successes_survive_a_failed_sibling_and_a_retry_resolves_it():
    from agent.execution_ledger import summarize

    agent = _agent()
    mail = "mcp__helpdesk__post_marketing_broadcasts_send"
    after_execute(agent, TAG, ARGS, "c1", _mcp({"status": "OK"}))
    after_execute(agent, mail, {"id": "b1"}, "c2", '{"error": "503 Service Unavailable"}')
    state = summarize(agent._execution_ledger.receipts)
    assert [r.operation_id for r in state[SUCCEEDED]] == ["c1"]
    assert [r.operation_id for r in state[UNKNOWN]] == ["c2"]
    # The successful write is refused on replay; only the unresolved one is worked on.
    assert before_execute(agent, TAG, ARGS, "c3") is not None
    before_execute(agent, "mcp__helpdesk__get_marketing_broadcasts_by_id", {"id": "b1"}, "r1")
    after_execute(agent, mail, {"id": "b1"}, "c4", _mcp({"status": "OK"}))
    state = summarize(agent._execution_ledger.receipts)
    assert not state[UNKNOWN] and not state[FAILED]


def test_ledger_resets_on_a_new_turn():
    agent = _agent("t1")
    after_execute(agent, TAG, ARGS, "c1", _mcp({"status": "OK"}))
    agent._current_turn_id = "t2"
    assert before_execute(agent, TAG, ARGS, "c2") is None
    assert ledger_for(agent).receipts == []


def test_scenario8_no_llm_or_network_call(monkeypatch):
    """Classification and the gate never call an LLM or the network."""
    import agent.oneshot
    import agent.auxiliary_client

    def boom(*_a, **_kw):
        raise AssertionError("no LLM call expected")

    monkeypatch.setattr(agent.oneshot, "run_oneshot", boom)
    monkeypatch.setattr(agent.auxiliary_client, "call_llm", boom)
    from agent.claim_stop import check_answer

    a = _agent()
    result = after_execute(a, TAG, ARGS, "c1", _mcp({"status": "OK"}))
    messages = [
        {"role": "user", "content": "gắn tag VIP"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": TAG, "arguments": json.dumps(ARGS)}}]},
        {"role": "tool", "tool_call_id": "c1", "content": result},
    ]
    assert check_answer(a, messages, "Đã gắn tag VIP.", 0) is None


def test_failures_fail_open(monkeypatch):
    monkeypatch.setattr(ledger_mod, "classify", lambda *_a, **_kw: 1 / 0)
    assert after_execute(_agent(), TAG, ARGS, "c1", "raw") == "raw"
