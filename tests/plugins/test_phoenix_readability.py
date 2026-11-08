"""Phoenix plugin readability: step/tool naming, tool unwrapping, error marking, judge spans."""

import importlib
import json
from types import SimpleNamespace

import pytest

phoenix = importlib.import_module("plugins.observability.phoenix")


def test_short_tool_and_dispatcher_unwrap():
    assert phoenix._short_tool("mcp__helpdesk__get_app_channel_by_type") == "helpdesk.get_app_channel_by_type"
    assert phoenix._short_tool("tool_search") == "tool_search"
    args = {"calls": [{"name": "mcp__helpdesk__get_x", "arguments": {"type": "ZALO_OA"}}]}
    assert phoenix._tool_targets("tool_call", args) == [("mcp__helpdesk__get_x", {"type": "ZALO_OA"})]
    assert phoenix._tool_label("tool_call", json.dumps(args)) == "helpdesk.get_x"
    assert phoenix._tool_targets("tool_search", {"queries": ["a"]}) == [("tool_search", {"queries": ["a"]})]


def test_tool_error_detection():
    assert phoenix._tool_error({"error": "ACTING user not linked"}) == "ACTING user not linked"
    assert phoenix._tool_error('{"status": "error", "message": "boom"}') == "boom"
    assert phoenix._tool_error({"result": '{"status": "OK", "data": []}'}) == ""
    assert phoenix._tool_error("plain text") == ""


def test_unwrap_result_parses_nested_json():
    raw = {"result": json.dumps({"status": "OK", "data": [{"config": json.dumps({"a": 1})}]})}
    assert phoenix._unwrap_result(raw) == {"status": "OK", "data": [{"config": {"a": 1}}]}


def test_decision_labels():
    call = SimpleNamespace(function=SimpleNamespace(
        name="tool_call", arguments=json.dumps({"calls": [{"name": "mcp__helpdesk__get_x", "arguments": {}}]})))
    label, text = phoenix._decision({"content": "\n\n", "reasoning": "need channels", "tool_calls": [call]})
    assert label == "→ helpdesk.get_x"
    assert text.startswith("Reasoning: need channels") and "→ helpdesk.get_x {}" in text
    label, text = phoenix._decision({"content": "Có 5 kênh.", "reasoning": None, "tool_calls": []})
    assert (label, text) == ("Trả lời", "Có 5 kênh.")


def test_input_digest_shows_new_messages_only():
    digest = phoenix._input_digest([
        {"role": "system", "content": "x" * 40},
        {"role": "user", "content": "Có bao nhiêu kênh?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "name": "mcp__helpdesk__get_x", "content": "{}"},
    ])
    assert digest == "[system prompt · 40 chars]\n\nuser:\nCó bao nhiêu kênh?\n\ntool helpdesk.get_x →\n{}"


# --- span tree (needs the OTel SDK; the Hermes image ships it) --------------------------------

@pytest.fixture
def spans(monkeypatch):
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(phoenix, "_tracer_for_project", lambda _project: tracer)
    monkeypatch.setitem(phoenix._TRACERS, "agent-chat", tracer)
    monkeypatch.setenv("HERMES_PHOENIX_CAPTURE", "full")
    phoenix._TRACE_STATE.clear()
    return exporter


def _msg(content, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [], reasoning=None)


def test_turn_tree(spans):
    ids = dict(task_id="t", session_id="s", turn_id="u", platform="tool")
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Có bao nhiêu kênh Zalo?"}]
    call = {"id": "c1", "function": {"name": "mcp__helpdesk__get_x", "arguments": "{}"}}

    phoenix.on_pre_llm_request(api_call_count=1, request_messages=history, system_prompt="sys", **ids)
    phoenix.on_post_llm_call(api_call_count=1, assistant_message=_msg("", [call]), **ids)
    phoenix.on_pre_tool_call(tool_name="mcp__helpdesk__get_x", args={}, tool_call_id="c1", **ids)
    phoenix.on_post_tool_call(tool_name="mcp__helpdesk__get_x", result={"error": "not linked"},
                              tool_call_id="c1", **ids)
    tool_row = {"role": "tool", "name": "mcp__helpdesk__get_x", "content": '{"error": "not linked"}'}
    phoenix.on_pre_llm_request(api_call_count=2, request_messages=history + [{"role": "assistant"}, tool_row], **ids)
    phoenix.on_post_llm_call(api_call_count=2, assistant_message=_msg("Chưa tra được."), **ids)
    phoenix.on_claim_gate(answer="Chưa tra được.", verdict="OK", receipts=[],
                          **{k: ids[k] for k in ("task_id", "session_id", "turn_id")})
    phoenix.on_post_llm_call(assistant_response="Chưa tra được.", **{k: ids[k] for k in ("task_id", "session_id", "turn_id")})

    by_name = {s.name: s for s in spans.get_finished_spans()}
    root = by_name["Có bao nhiêu kênh Zalo?"]
    step1, tool, step2 = by_name["1. → helpdesk.get_x"], by_name["helpdesk.get_x"], by_name["2. Trả lời"]
    gate = by_name["Claim gate: OK"]

    assert tool.parent.span_id == step1.context.span_id  # tool nests under the step that called it
    assert tool.status.status_code.name == "ERROR"
    assert tool.attributes["output.value"] == "ERROR: not linked"
    assert step2.parent.span_id == root.context.span_id
    assert "tool helpdesk.get_x" in step2.attributes["input.value"]  # only what is new since step 1
    assert "sys" not in step2.attributes["input.value"]
    assert gate.parent.span_id == root.context.span_id
    assert gate.attributes["output.value"] == "OK — answer accepted"
    assert root.attributes["output.value"] == "Chưa tra được."
    assert root.attributes["hermes.summary"].startswith("2 bước · 1 tool (1 lỗi)")


def test_subagent_turn_nests_inside_the_parent_trace(spans):
    ids = dict(task_id="t", session_id="s", turn_id="u", platform="tool")
    turn = {k: ids[k] for k in ("task_id", "session_id", "turn_id")}
    history = [{"role": "user", "content": "Lấy khách theo tag"}]
    call = {"id": "d1", "function": {"name": "delegate_task", "arguments": "{}"}}
    phoenix._SUBAGENT_PARENTS.clear()

    phoenix.on_pre_llm_request(api_call_count=1, request_messages=history, **ids)
    phoenix.on_post_llm_call(api_call_count=1, assistant_message=_msg("", [call]), **ids)
    phoenix.on_pre_tool_call(tool_name="delegate_task", args={}, tool_call_id="d1", **ids)
    phoenix.on_subagent_start(parent_turn_id="u", child_session_id="child", child_role="leaf",
                              child_goal="Tra khách cho tag VIP")
    # The child runs its own turn under its own session id and platform.
    child = dict(task_id="sa-0", session_id="child", turn_id="child:sa-0", platform="subagent")
    phoenix.on_pre_llm_request(api_call_count=1, request_messages=[{"role": "user", "content": "Tra khách cho tag VIP"}],
                               **child)
    phoenix.on_post_llm_call(api_call_count=1, assistant_message=_msg("Không có khách."), **child)
    phoenix.on_post_llm_call(assistant_response="Không có khách.",
                             **{k: child[k] for k in ("task_id", "session_id", "turn_id")})
    phoenix.on_subagent_stop(parent_turn_id="u", child_session_id="child", child_role="leaf",
                             child_summary="Không có khách.", child_status="completed")
    phoenix.on_post_tool_call(tool_name="delegate_task", result={"results": []}, tool_call_id="d1", **ids)
    phoenix.on_post_llm_call(assistant_response="Xong.", **turn)

    by_name = {s.name: s for s in spans.get_finished_spans()}
    root, delegate = by_name["Lấy khách theo tag"], by_name["delegate_task"]
    agent, child_root = by_name["Subagent: Tra khách cho tag VIP"], by_name["Tra khách cho tag VIP"]
    assert agent.parent.span_id == delegate.context.span_id  # subagents nest under the delegate call
    assert child_root.parent.span_id == agent.context.span_id  # the child's turn nests under its subagent
    assert child_root.context.trace_id == root.context.trace_id  # one trace, not five separate ones
    assert by_name["1. Trả lời"].parent.span_id == child_root.context.span_id
    assert not phoenix._SUBAGENT_PARENTS


# --- full input mode: every step shows the whole request ------------------------------------

def _full_request():
    bridged = {"id": "c1", "type": "function", "function": {
        "name": "tool_call",
        "arguments": json.dumps({"calls": [{"name": "mcp__helpdesk__search_customers", "arguments": {}}]})}}
    return [
        {"role": "system", "content": "SYS" * 10},
        {"role": "user", "content": "cho tôi danh sách khách hàng"},
        {"role": "assistant", "content": "", "tool_calls": [bridged], "reasoning_content": "need the list"},
        {"role": "tool", "tool_call_id": "c1", "content": "[Tool output from an earlier turn cleared to save context.]"},
        {"role": "assistant", "content": "Có 597 khách.", "reasoning_content": " "},
        {"role": "user", "content": "có bao nhiêu pipeline?"},
    ]


def test_full_input_view_splits_system_history_and_this_turn():
    view = phoenix._full_input_view(_full_request(), new_from=5)
    assert list(view) == ["overview", "system_prompt", "conversation_history", "this_turn"]
    assert view["system_prompt"] == "SYS" * 10
    assert view["overview"]["history_turns"] == 1 and view["overview"]["new_since_previous_step"] == [5]
    assert view["overview"]["history_reasoning_chars"] == len("need the list")  # the " " pad counts 0
    turn = view["conversation_history"][0]
    assert [list(m)[1] for m in turn["messages"]] == ["user", "assistant", "tool_result", "assistant"]
    call = turn["messages"][1]
    assert call["tool_calls"] == [{"tool": "helpdesk.search_customers", "arguments": {}}]
    assert call["reasoning"] == "need the list"
    assert turn["messages"][2]["tool_result"] == "helpdesk.search_customers"
    assert "reasoning" not in turn["messages"][3]  # a blank pad is not shown as reasoning
    assert view["this_turn"] == [{"#": 5, "new": True, "user": "có bao nhiêu pipeline?"}]


def test_full_input_mode_shows_whole_request_every_step(spans, monkeypatch):
    monkeypatch.setenv("HERMES_PHOENIX_INPUT", "full")
    ids = dict(task_id="t2", session_id="s2", turn_id="u2", platform="tool")
    request = _full_request()
    phoenix.on_pre_llm_request(api_call_count=1, request_messages=request, **ids)
    phoenix.on_post_llm_call(api_call_count=1, assistant_message=_msg("2 pipeline."), **ids)
    phoenix.on_post_llm_call(assistant_response="2 pipeline.", **{k: ids[k] for k in ("task_id", "session_id", "turn_id")})

    step = next(s for s in spans.get_finished_spans() if s.name.startswith("1."))
    attrs = step.attributes
    assert attrs["llm.input_messages.0.message.role"] == "system"
    assert attrs["llm.input_messages.5.message.content"] == "có bao nhiêu pipeline?"  # all 6 sent rows
    assert attrs["llm.input_messages.3.message.tool_call_id"] == "c1"
    assert attrs["llm.input_messages.3.message.name"] == "helpdesk.search_customers"
    assert attrs["llm.input_messages.2.message.content"].startswith("[reasoning_content · 13 chars]\nneed the list")
    assert attrs["input.mime_type"] == "application/json"
    view = json.loads(attrs["input.value"])
    assert view["conversation_history"][0]["messages"][0]["user"] == "cho tôi danh sách khách hàng"
    assert view["this_turn"][0]["user"] == "có bao nhiêu pipeline?"


def test_delta_input_mode_is_the_default(spans, monkeypatch):
    monkeypatch.delenv("HERMES_PHOENIX_INPUT", raising=False)
    ids = dict(task_id="t3", session_id="s3", turn_id="u3", platform="tool")
    phoenix.on_pre_llm_request(api_call_count=1, request_messages=_full_request(), system_prompt="SYS", **ids)
    phoenix.on_post_llm_call(api_call_count=1, assistant_message=_msg("ok"), **ids)
    phoenix.on_post_llm_call(assistant_response="ok", **{k: ids[k] for k in ("task_id", "session_id", "turn_id")})
    step = next(s for s in spans.get_finished_spans() if s.name.startswith("1."))
    assert "llm.input_messages.2.message.role" not in step.attributes  # system + this turn's user only
