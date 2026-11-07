"""A reconnect re-lists tools and applies changes; a safety-net re-list runs on the keepalive cadence."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_health import _tools_fingerprint
from tools.mcp_tool_registration import _register_server_tools
from tools.registry import ToolRegistry


def _tool(name: str):
    return SimpleNamespace(name=name, description="", inputSchema=None)


def _connected_server(name: str, registry: ToolRegistry, tools: list) -> MCPServerTask:
    server = MCPServerTask(name)
    server._config = {"url": "https://example.test/mcp"}
    server._tools = tools
    server.session = MagicMock()
    with patch("tools.registry.registry", registry):
        server._registered_tool_names = _register_server_tools(name, server, server._config)
    return server


class TestReconnectRelist:
    @pytest.mark.asyncio
    async def test_reconnect_applies_tool_added_during_outage(self):
        registry = ToolRegistry()
        server = _connected_server("hd", registry, [_tool("old_tool")])
        server.session = SimpleNamespace(
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[_tool("old_tool"), _tool("new_tool")])))

        with patch("tools.registry.registry", registry):
            await server._discover_tools()

        assert "mcp__hd__new_tool" in registry.get_all_tool_names()
        assert {"mcp__hd__new_tool", "mcp__hd__old_tool"} <= set(server._registered_tool_names)

    @pytest.mark.asyncio
    async def test_reconnect_drops_tool_removed_during_outage(self):
        registry = ToolRegistry()
        server = _connected_server("hd", registry, [_tool("keep"), _tool("gone")])
        server.session = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(tools=[_tool("keep")])))

        with patch("tools.registry.registry", registry):
            await server._discover_tools()

        assert "mcp__hd__gone" not in registry.get_all_tool_names()
        assert "mcp__hd__keep" in server._registered_tool_names
        assert "mcp__hd__gone" not in server._registered_tool_names

    @pytest.mark.asyncio
    async def test_unchanged_relist_does_not_reregister(self):
        registry = ToolRegistry()
        server = _connected_server("hd", registry, [_tool("same")])
        before = list(server._registered_tool_names)
        server.session = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(tools=[_tool("same")])))

        with patch("tools.registry.registry", registry), \
                patch("tools.mcp_tool_registration._register_server_tools") as register:
            await server._discover_tools()

        register.assert_not_called()
        assert server._registered_tool_names == before
        assert server._tools_listed_at > 0


    @pytest.mark.asyncio
    async def test_reconnect_registers_utilities_for_newly_advertised_capability(self):
        registry = ToolRegistry()
        server = _connected_server("hd", registry, [_tool("same")])
        server.initialize_result = _caps(resources=None)
        with patch("tools.registry.registry", registry):
            server._registered_tool_names = _register_server_tools("hd", server, server._config)
        assert "mcp__hd__list_resources" not in server._registered_tool_names

        server.initialize_result = _caps(resources=SimpleNamespace())
        server.session = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(tools=[_tool("same")])))
        with patch("tools.registry.registry", registry):
            await server._discover_tools()

        assert "mcp__hd__list_resources" in server._registered_tool_names

    @pytest.mark.asyncio
    async def test_reconnect_applies_description_change_under_same_name(self):
        from mcp.types import Tool

        schema = {"type": "object", "properties": {}}
        registry = ToolRegistry()
        server = _connected_server("hd", registry, [Tool(name="search", description="old", inputSchema=schema)])
        server.session = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(
            tools=[Tool(name="search", description="new", inputSchema=schema)])))

        with patch("tools.registry.registry", registry):
            await server._discover_tools()
            schema_desc = registry.get_schema("mcp__hd__search")["description"]

        assert "new" in schema_desc


def _caps(resources=None, prompts=None):
    return SimpleNamespace(capabilities=SimpleNamespace(tools=SimpleNamespace(), resources=resources, prompts=prompts))


class TestToolsRelistInterval:
    def _server(self, config: dict) -> MCPServerTask:
        server = MCPServerTask("hd")
        server._config = config
        return server

    def test_default_when_unset(self):
        from tools.mcp_tool import _DEFAULT_TOOLS_RELIST_INTERVAL
        assert self._server({"url": "https://x/mcp"})._tools_relist_interval() == _DEFAULT_TOOLS_RELIST_INTERVAL

    def test_zero_disables(self):
        assert self._server({"url": "https://x/mcp", "tools_relist_interval": 0})._tools_relist_interval() is None

    def test_below_floor_is_clamped(self):
        from tools.mcp_tool import _MIN_TOOLS_RELIST_INTERVAL
        interval = self._server({"url": "https://x/mcp", "tools_relist_interval": 1})._tools_relist_interval()
        assert interval == _MIN_TOOLS_RELIST_INTERVAL

    @pytest.mark.parametrize("bad", ["not-a-number", "nan", "inf", float("nan"), float("inf")])
    def test_invalid_falls_back_to_default_with_warning(self, bad, caplog):
        from tools.mcp_tool import _DEFAULT_TOOLS_RELIST_INTERVAL
        with caplog.at_level("WARNING", logger="tools.mcp_tool"):
            interval = self._server({"url": "https://x/mcp", "tools_relist_interval": bad})._tools_relist_interval()
        assert interval == _DEFAULT_TOOLS_RELIST_INTERVAL
        assert sum("invalid tools_relist_interval" in r.message for r in caplog.records) == 1

    def test_stdio_never_relists(self):
        assert self._server({"command": "srv"})._tools_relist_interval() is None

    def test_never_due_before_first_listing(self):
        server = self._server({"url": "https://x/mcp"})
        assert server._tools_relist_due(server._tools_relist_interval()) is False


class TestKeepaliveSchedulesRelist:
    async def _one_keepalive_cycle(self, task):
        real_wait = asyncio.wait
        cycles = {"n": 0}

        async def fake_wait(tasks, timeout=None, return_when=None):
            cycles["n"] += 1
            if cycles["n"] == 1:
                return set(), set(tasks)
            task._shutdown_event.set()
            return await real_wait(tasks, timeout=0.5, return_when=return_when or asyncio.FIRST_COMPLETED)

        import tools.mcp_tool as mcp_mod
        orig = mcp_mod.asyncio.wait
        mcp_mod.asyncio.wait = fake_wait
        try:
            return await task._wait_for_lifecycle_event()
        finally:
            mcp_mod.asyncio.wait = orig

    def _task(self, listed_ago: float) -> MCPServerTask:
        task = MCPServerTask("hd")
        task._config = {"url": "https://x/mcp"}
        task.session = SimpleNamespace(list_tools=AsyncMock(), send_ping=AsyncMock())
        task._tools_listed_at = time.monotonic() - listed_ago
        return task

    @pytest.mark.asyncio
    async def test_due_relist_is_scheduled_after_healthy_keepalive(self):
        task = self._task(listed_ago=10_000)
        with patch.object(MCPServerTask, "_schedule_tools_refresh") as schedule:
            assert await self._one_keepalive_cycle(task) == "shutdown"
        schedule.assert_called_once()

    @pytest.mark.asyncio
    async def test_relist_not_scheduled_before_interval(self):
        task = self._task(listed_ago=1)
        with patch.object(MCPServerTask, "_schedule_tools_refresh") as schedule:
            assert await self._one_keepalive_cycle(task) == "shutdown"
        schedule.assert_not_called()


class TestToolsFingerprint:
    def test_description_change_under_same_name_is_detected(self):
        from mcp.types import Tool

        schema = {"type": "object", "properties": {}}
        before = [Tool(name="search", description="old", inputSchema=schema)]
        after = [Tool(name="search", description="new", inputSchema=schema)]
        assert _tools_fingerprint(before) != _tools_fingerprint(after)

    def test_order_does_not_matter(self):
        from mcp.types import Tool

        schema = {"type": "object", "properties": {}}
        a = Tool(name="a", description="", inputSchema=schema)
        b = Tool(name="b", description="", inputSchema=schema)
        assert _tools_fingerprint([a, b]) == _tools_fingerprint([b, a])


class TestRefreshTimeout:
    @pytest.mark.asyncio
    async def test_stalled_list_releases_lock_and_keeps_the_session(self):
        async def never_returns(*_args, **_kwargs):
            await asyncio.Event().wait()

        server = MCPServerTask("hd")
        server._config = {"url": "https://x/mcp"}
        server.session = SimpleNamespace(list_tools=never_returns)
        with patch("tools.mcp_tool._TOOLS_LIST_TIMEOUT", 0.05):
            await server._refresh_tools()

        assert server._rpc_lock.locked() is False
        assert not server._reconnect_event.is_set()
        assert server._suspect_reason is None

    @pytest.mark.asyncio
    async def test_timeout_after_reconnect_leaves_fresh_session_alone(self):
        server = MCPServerTask("hd")
        server._config = {"url": "https://x/mcp"}
        fresh = SimpleNamespace(list_tools=AsyncMock())

        async def swap_then_stall(*_args, **_kwargs):
            server.session = fresh
            await asyncio.Event().wait()

        server.session = SimpleNamespace(list_tools=swap_then_stall)
        with patch("tools.mcp_tool._TOOLS_LIST_TIMEOUT", 0.05):
            await server._refresh_tools()

        assert server.session is fresh
        assert not server._reconnect_event.is_set()


class TestScheduleRelistIfDue:
    def _server(self, listed_ago: float) -> MCPServerTask:
        server = MCPServerTask("hd")
        server._config = {"url": "https://x/mcp"}
        server._tools_listed_at = time.monotonic() - listed_ago
        server._advertises_tools = lambda: True
        server._schedule_tools_refresh = MagicMock()
        return server

    def test_schedules_when_due(self):
        server = self._server(listed_ago=400)
        server._schedule_relist_if_due(300)
        server._schedule_tools_refresh.assert_called_once()

    def test_skips_when_not_due(self):
        server = self._server(listed_ago=10)
        server._schedule_relist_if_due(300)
        server._schedule_tools_refresh.assert_not_called()

    def test_skips_while_a_refresh_is_pending(self):
        server = self._server(listed_ago=400)
        server._pending_refresh_tasks.add(object())
        server._schedule_relist_if_due(300)
        server._schedule_tools_refresh.assert_not_called()


class TestFailedRefreshBacksOff:
    """A skipped or failed refresh counts as an attempt: the next re-list waits a full interval
    instead of re-firing (and re-taking the RPC lock) on every keepalive wake."""

    def _server(self, listed_ago: float) -> MCPServerTask:
        server = MCPServerTask("hd")
        server._config = {"url": "https://x/mcp"}
        server._tools_listed_at = time.monotonic() - listed_ago
        server._advertises_tools = lambda: True
        return server

    @pytest.mark.asyncio
    async def test_timed_out_relist_is_not_due_again_immediately(self):
        async def never_returns(*_args, **_kwargs):
            await asyncio.Event().wait()

        server = self._server(listed_ago=400)
        server.session = SimpleNamespace(list_tools=never_returns)
        assert server._tools_relist_due(300)
        with patch("tools.mcp_tool._TOOLS_LIST_TIMEOUT", 0.05):
            await server._refresh_tools()

        assert not server._tools_relist_due(300)

    @pytest.mark.asyncio
    async def test_relist_skipped_without_session_is_not_due_again_immediately(self):
        server = self._server(listed_ago=400)
        server.session = None
        await server._refresh_tools()

        assert not server._tools_relist_due(300)

    @pytest.mark.asyncio
    async def test_failed_relist_is_not_due_again_immediately(self):
        server = self._server(listed_ago=400)
        server.session = SimpleNamespace(list_tools=AsyncMock(side_effect=RuntimeError("boom")))
        with pytest.raises(RuntimeError):
            await server._refresh_tools()

        assert not server._tools_relist_due(300)

    @pytest.mark.asyncio
    async def test_failed_refresh_never_arms_relist_before_first_listing(self):
        server = self._server(listed_ago=0)
        server._tools_listed_at = 0.0
        server.session = None
        await server._refresh_tools()

        assert server._tools_listed_at == 0.0


class TestAdoptersRefreshedAfterOwnerState:
    @pytest.mark.asyncio
    async def test_owner_registered_names_are_final_when_adopters_refresh(self):
        registry = ToolRegistry()
        server = _connected_server("hd", registry, [_tool("old_tool")])
        server.session = SimpleNamespace(
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[_tool("old_tool"), _tool("new_tool")])))
        seen = []

        def record(name, srv):
            seen.append(set(srv._registered_tool_names))

        with patch("tools.registry.registry", registry), \
                patch("tools.mcp_tool_lifecycle.refresh_adopters", side_effect=record):
            await server._discover_tools()

        assert seen and "mcp__hd__new_tool" in seen[0]
