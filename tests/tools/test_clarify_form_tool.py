"""Tests for tools/clarify_form_tool.py - answer shapes returned by platform callbacks."""

import json
from unittest.mock import patch

import pytest

from tools.clarify_form_tool import clarify_form_tool

QUESTIONS = [
    {"question": "Which platform?", "choices": ["MESSENGER", "EMAIL"]},
    {"question": "Send when?"},
]


def _run(callback_result):
    return json.loads(clarify_form_tool(QUESTIONS, callback=lambda _qs: callback_result))


@pytest.mark.parametrize(
    "callback_result",
    [
        [{"id": "q1", "answer": "EMAIL"}, {"id": "q2", "answer": "now"}],
        {"q1": "EMAIL", "q2": "now"},
        json.dumps({"q1": "EMAIL", "q2": "now"}),
        json.dumps([{"id": "q1", "answer": "EMAIL"}, {"id": "q2", "answer": "now"}]),
    ],
)
def test_every_answer_shape_is_read(callback_result):
    result = _run(callback_result)
    assert result["answered"] is True
    assert result["responses"] == [
        {"question": "Which platform?", "answer": "EMAIL"},
        {"question": "Send when?", "answer": "now"},
    ]


@pytest.mark.parametrize("callback_result", [None, "", "not json", 42])
def test_cancel_or_garbage_is_unanswered(callback_result):
    assert _run(callback_result) == {"responses": [], "answered": False}


def test_dashboard_bridge_round_trip():
    """The tui_gateway bridge result must be readable by the tool (regression: it
    returned a JSON string the tool treated as unanswered, so the agent re-asked)."""
    from tui_gateway import server

    with patch("tui_gateway.server_requests.send", return_value={"answers": {"q1": "EMAIL", "q2": "now"}}), \
            patch.object(server, "_clarify_timeout_seconds", return_value=1):
        result = json.loads(clarify_form_tool(
            QUESTIONS, callback=lambda qs: server._clarify_form_block("sid", qs)))

    assert result["answered"] is True
    assert [r["answer"] for r in result["responses"]] == ["EMAIL", "now"]


def test_dashboard_bridge_cancel():
    from tui_gateway import server

    with patch("tui_gateway.server_requests.send", return_value=None), \
            patch.object(server, "_clarify_timeout_seconds", return_value=1):
        assert server._clarify_form_block("sid", []) is None
