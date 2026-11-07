"""A re-list by the owner of a shared MCP connection must reach the profiles that adopted it."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from tests.tools.test_mcp_multiplex_connection_keys import _server, two_profiles  # noqa: F401


def _named_tool(name):
    return SimpleNamespace(name=name, description="d", inputSchema={"type": "object", "properties": {}},
                           annotations=None)


def _owner_and_adopter(two_profiles, cfg):  # noqa: F811
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    two_profiles("a")
    srv = _server("x", cfg)
    disc._adopt_server("x", srv)
    srv._registered_tool_names = reg._register_server_tools("x", srv, cfg)
    two_profiles("b")
    assert reg.register_connected_into_current_scope({"x": cfg}) == 1
    return srv


def _relist_as_owner(two_profiles, srv, cfg, adopter_cfg):  # noqa: F811
    from tools import mcp_tool_lifecycle as lifecycle
    from tools import mcp_tool_registration as reg

    two_profiles("a")
    srv._tools = [_named_tool("t"), _named_tool("t2")]
    srv._registered_tool_names = reg._register_server_tools("x", srv, cfg)
    with patch("tools.mcp_tool_config._load_mcp_config", lambda: {"x": adopter_cfg}):
        lifecycle.refresh_adopters("x", srv)


def test_adopting_profile_receives_a_tool_the_owner_relisted(two_profiles):  # noqa: F811
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer shared"}}
    srv = _owner_and_adopter(two_profiles, cfg)
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]

    _relist_as_owner(two_profiles, srv, cfg, cfg)

    two_profiles("b")
    assert sorted(registry.get_tool_names_for_toolset("mcp-x")) == ["mcp__x__t", "mcp__x__t2"]


def test_adopter_keeps_its_own_tool_filter(two_profiles):  # noqa: F811
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer shared"}}
    srv = _owner_and_adopter(two_profiles, cfg)

    _relist_as_owner(two_profiles, srv, cfg, {**cfg, "tools": {"exclude": ["t2"]}})

    two_profiles("a")
    assert sorted(registry.get_tool_names_for_toolset("mcp-x")) == ["mcp__x__t", "mcp__x__t2"]
    two_profiles("b")
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]
