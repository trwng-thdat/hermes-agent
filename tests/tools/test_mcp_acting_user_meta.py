"""A ``forward_session_user`` MCP server receives the chatting user's SSO account id as
``_meta["ai.jarvis/actingAccountId"]`` on tools/call; calls without a real end user carry no meta."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.session_context import reset_session_vars, set_session_vars
from tools import mcp_tool
from tools import mcp_tool_handlers as _mcp_handlers

_META_KEY = "ai.jarvis/actingAccountId"


def _run_inline(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    loop = asyncio.new_event_loop()
    try:
        async def _with_lock():
            for srv in list(mcp_tool._servers.values()):
                if getattr(srv, "_rpc_lock", None) is None:
                    srv._rpc_lock = asyncio.Lock()
            return await coro
        return loop.run_until_complete(_with_lock())
    finally:
        loop.close()


def _result():
    return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], isError=False,
                           structuredContent=None, meta=None)


@pytest.fixture
def call(monkeypatch):
    """``call(server_name, config, user_id=..., account_id=...) -> (args, kwargs)`` of session.call_tool."""
    for name in ("HERMES_SESSION_USER_ID", "HERMES_SESSION_ACTING_ACCOUNT_ID"):
        monkeypatch.delenv(name, raising=False)

    def _call(server_name, config, *, user_id=None, account_id=""):
        session = MagicMock()
        session.call_tool = AsyncMock(return_value=_result())
        server = SimpleNamespace(session=session, _rpc_lock=None, _config=config)
        reset_session_vars()
        if user_id is not None:
            set_session_vars(user_id=user_id, acting_account_id=account_id)
        with patch.dict(mcp_tool._servers, {server_name: server}), \
             patch("tools.mcp_tool_loop._run_on_mcp_loop", side_effect=_run_inline):
            _mcp_handlers._make_tool_handler(server_name, "list_tickets", 30.0)({"status": "open"})
        session.call_tool.assert_awaited_once()
        return session.call_tool.await_args

    yield _call
    reset_session_vars()


FORWARD = {"url": "https://helpdesk.example/mcp", "forward_session_user": True}


def test_forwarding_server_gets_the_session_account_id(call):
    args, kwargs = call("helpdesk", FORWARD, user_id="u-1", account_id="hd-42")
    assert kwargs["meta"] == {_META_KEY: "hd-42"}
    assert kwargs["arguments"] == {"status": "open"}


def test_unlinked_user_sends_empty_account_id_not_no_meta(call):
    _, kwargs = call("helpdesk", FORWARD, user_id="u-1", account_id="")
    assert kwargs["meta"] == {_META_KEY: ""}


@pytest.mark.parametrize("user_id", [None, "", "system"])
def test_no_real_user_sends_no_meta(call, user_id):
    _, kwargs = call("helpdesk", FORWARD, user_id=user_id, account_id="hd-42")
    assert "meta" not in kwargs


@pytest.mark.parametrize("config", [{"url": "https://x/mcp"}, {"url": "https://x/mcp", "forward_session_user": False}])
def test_server_without_the_flag_sends_no_meta(call, config):
    _, kwargs = call("helpdesk", config, user_id="u-1", account_id="hd-42")
    assert "meta" not in kwargs


def test_jarvis_mcp_still_gets_session_user_id_argument(call):
    _, kwargs = call("jarvis-mcp", {"url": "https://x/mcp"}, user_id="u-1", account_id="hd-42")
    assert kwargs["arguments"] == {"status": "open", "user_id": "u-1"}
    assert "meta" not in kwargs
