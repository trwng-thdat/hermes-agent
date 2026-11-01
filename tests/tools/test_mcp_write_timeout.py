"""A write-capable MCP tools/call that times out after dispatch reports an UNKNOWN outcome
(``outcome_uncertain``) instead of a plain failure, and is never replayed."""

import json
from types import SimpleNamespace

from tools import mcp_tool_handlers as handlers


def test_write_timeout_is_outcome_uncertain(monkeypatch):
    calls = []

    def run_on_loop(_call, timeout=None):
        calls.append(timeout)
        raise TimeoutError("MCP call timed out after 30.0s (configured timeout: 30.0s)")

    monkeypatch.setattr(handlers._loop, "_run_on_mcp_loop", run_on_loop)
    monkeypatch.setattr("tools.mcp_tool._bump_server_error", lambda *_a, **_kw: None)
    result = handlers._dispatch(
        "helpdesk", SimpleNamespace(), "tools/call post_customer_by_customer_id_tags", lambda: None, 30.0,
        (handlers._handle_write_timeout,), lambda _exc: None,
    )
    data = json.loads(result)
    assert data["outcome_uncertain"] is True
    assert "UNKNOWN" in data["error"] and "NOT retried" in data["error"]
    assert calls == [30.0]  # dispatched once, never replayed


def test_read_timeout_stays_a_plain_failure(monkeypatch):
    def run_on_loop(_call, timeout=None):
        raise TimeoutError("MCP call timed out after 30.0s")

    monkeypatch.setattr(handlers._loop, "_run_on_mcp_loop", run_on_loop)
    result = handlers._dispatch(
        "helpdesk", SimpleNamespace(), "tools/call get_customer", lambda: None, 30.0, (), lambda _exc: None,
    )
    data = json.loads(result)
    assert "outcome_uncertain" not in data and "timed out" in data["error"]


def test_write_timeout_handler_ignores_other_errors():
    assert handlers._handle_write_timeout("helpdesk", RuntimeError("boom"), lambda: None, "op") is None
