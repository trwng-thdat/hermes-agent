"""A reconnect re-lists tools and applies changes; a safety-net re-list runs on the keepalive cadence."""

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
