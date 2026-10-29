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
    phoenix.on_claim_gate(answer="Chưa tra được.", verdict="OK", judge="ok", judge_input="ANSWER: …",
                          judge_reply='{"verdict": "OK", "issues": []}', judge_started=1.0, judge_ended=2.0,
                          **{k: ids[k] for k in ("task_id", "session_id", "turn_id")})
    phoenix.on_post_llm_call(assistant_response="Chưa tra được.", **{k: ids[k] for k in ("task_id", "session_id", "turn_id")})

    by_name = {s.name: s for s in spans.get_finished_spans()}
    root = by_name["Có bao nhiêu kênh Zalo?"]
    step1, tool, step2 = by_name["1. → helpdesk.get_x"], by_name["helpdesk.get_x"], by_name["2. Trả lời"]
    gate, judge = by_name["Claim gate: OK"], by_name["Judge"]

    assert tool.parent.span_id == step1.context.span_id  # tool nests under the step that called it
    assert tool.status.status_code.name == "ERROR"
    assert tool.attributes["output.value"] == "ERROR: not linked"
    assert step2.parent.span_id == root.context.span_id
    assert "tool helpdesk.get_x" in step2.attributes["input.value"]  # only what is new since step 1
    assert "sys" not in step2.attributes["input.value"]
    assert judge.parent.span_id == gate.context.span_id
    assert json.loads(judge.attributes["output.value"]) == {"verdict": "OK", "issues": []}
    assert root.attributes["output.value"] == "Chưa tra được."
    assert root.attributes["hermes.summary"].startswith("2 bước · 1 tool (1 lỗi)")
