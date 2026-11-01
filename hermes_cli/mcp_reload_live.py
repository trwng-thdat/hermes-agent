"""Bring one MCP server up to date in THIS process for the caller's profile, without a restart.

A backend that rewrites ``mcp_servers.<name>`` in a profile's config (an integration connected,
its token rotated, or disconnected) calls :func:`reload_mcp_server` under that profile's home and
secret scope (``POST /api/mcp/servers/{name}/reload``). Unlike ``reload.mcp``, which tears down every
server of every profile, it touches only ``name`` and only this profile's side of it:

- removed or ``enabled: false``: close the connection this profile owns, drop its overlay on a
  shared connection it adopted, and hand any orphaned adopters back their tools;
- configured, and the live connection this profile uses has the same identity (route plus
  credentials): no teardown, only ``register_mcp_servers`` so the overlay heals idempotently;
- configured with another identity, or not connected: close this profile's connection, register.

Several profiles commonly share ONE live connection (same URL and token): the first profile owns
it, the others adopt it. Tearing it down needlessly cuts the adopters' tools mid-turn, hence the
identity check; when it must go, ``register_mcp_servers`` re-registers the orphaned adopters
before returning (``mcp_tool_lifecycle._reregister_orphaned_adopters``).
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_reload_lock = threading.Lock()


def reload_mcp_server(name: str) -> Dict[str, Any]:
    """Reload ``name`` for the caller's profile and refresh that profile's open chats.

    Returns ``{name, connected, tools, removed, reconnected, error?}``. A server that fails to
    connect is reported with ``connected: False``; never raises."""
    row: Dict[str, Any] = {"name": name, "connected": False, "tools": [], "removed": False,
                           "reconnected": False}
    before = _visible_tools(name)
    try:
        with _registry_reload_lock():
            row.update(_reload(name))
    except Exception as exc:
        logger.warning("MCP reload of %r failed", name, exc_info=True)
        row["error"] = str(exc) or type(exc).__name__
    if row["reconnected"] or set(row["tools"]) != set(before):
        _refresh_open_chats(name, row)
    return row


def _reload(name: str) -> Dict[str, Any]:
    from tools.mcp_oauth import suppress_interactive_oauth
    from tools.mcp_tool import _mcp_registry_scope
    from tools.mcp_tool_common import mcp_server_enabled
    from tools.mcp_tool_config import _load_mcp_config
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers

    scope = _mcp_registry_scope()
    config = _load_mcp_config().get(name)
    if config is None and not _absent_from_config_file(name):
        # _load_mcp_config answers {} on any read error: never mistake that for "removed".
        raise RuntimeError("the profile's MCP config could not be read; nothing was changed")
    # No browser/stdin OAuth prompt from a backend call; a server needing one reports not connected.
    if not isinstance(config, dict) or not mcp_server_enabled(config):
        with suppress_interactive_oauth():
            _remove(name, scope)
        return {"removed": True}
    reconnect = not _live_connection_matches(name, scope, config)
    with suppress_interactive_oauth():
        if reconnect:
            shutdown_mcp_servers(scope=scope, names={name})
            _forget_own_lazy(name, scope)
        register_mcp_servers({name: config})
    from hermes_cli.plugins_activation_live import _server_error, _utility_suffixes
    from tools.connectors.mcp import _registered_tool_names
    tools = [t for t in _registered_tool_names(name) if not t.endswith(_utility_suffixes())]
    result: Dict[str, Any] = {"connected": bool(tools) or _has_live_session(name, scope),
                              "tools": tools, "reconnected": reconnect}
    if not result["connected"]:
        result["error"] = _server_error(name) or "the server did not connect"
    return result


def _registry_reload_lock() -> threading.Lock:
    """The chat host's ``reload.mcp`` lock when this process hosts chats, so a ``/reload-mcp`` and
    this reload never interleave teardown and registration; this module's own lock otherwise."""
    server = sys.modules.get("tui_gateway.server")
    return getattr(server, "_mcp_reload_lock", None) or _reload_lock


def _absent_from_config_file(name: str) -> bool:
    """Whether ``name`` is really gone from this profile's config.yaml, read raw: the interpolating
    loader also drops an entry it failed to expand, and that is not a removal."""
    from hermes_cli.config import load_config
    from utils import env_var_enabled

    if env_var_enabled("HERMES_SAFE_MODE"):
        return False
    try:
        servers = load_config().get("mcp_servers")
    except Exception:
        return False
    return not isinstance(servers, dict) or name not in servers


def _remove(name: str, scope: Optional[str]) -> None:
    """Close the connection ``scope`` owns for ``name`` and drop ``scope``'s overlay on a shared
    one it adopted. Only ``name`` is touched: the overlay is removed directly instead of through
    ``register_mcp_servers({})``, whose healing pass re-judges every adopted server of the profile."""
    from tools import mcp_tool as core
    from tools.mcp_tool_lifecycle import _reregister_orphaned_adopters, shutdown_mcp_servers
    from tools.mcp_tool_registration import _remove_server_scope
    from tools.mcp_tool_scope import _key_name

    shutdown_mcp_servers(scope=scope, names={name})
    _forget_own_lazy(name, scope)
    if scope is not None:
        with core._lock:
            adopted = [key for key, scopes in core._server_tool_scopes.items()
                       if scope in scopes and _key_name(key) == name]
        for key in adopted:
            _remove_server_scope(key, scope)
    # Profiles that adopted the connection just closed get their own (or nothing, per their config).
    _reregister_orphaned_adopters()


def _live_connection_matches(name: str, scope: Optional[str], config: dict) -> bool:
    """Whether the connection ``scope`` uses for ``name`` (its own, or one it adopted) is live and
    has ``config``'s identity. A lazily registered server (schema cache, not spawned yet) matches
    on its stored config. Cross-profile OAuth never matches (``_same_server_route``)."""
    from tools import mcp_tool as core
    from tools.mcp_tool_registration import _connection_identity, _same_server_route
    from tools.mcp_tool_scope import _key_scope, _resolve_server_key

    with core._lock:
        key = _resolve_server_key(name, scope, current=False)
        server = core._servers.get(key)
        lazy = core._lazy_server_configs.get(key)
    if server is not None:
        return (getattr(server, "session", None) is not None
                and _same_server_route(server, config, cross_profile=_key_scope(key) != scope))
    return lazy is not None and _connection_identity(lazy) == _connection_identity(config)


def _forget_own_lazy(name: str, scope: Optional[str]) -> None:
    """Drop this profile's schema-cache registration: it would connect with the old config."""
    from tools import mcp_tool as core
    from tools.mcp_tool_discovery import _forget_lazy_server
    from tools.mcp_tool_scope import _server_key

    key = _server_key(name, scope, current=False)
    if key in core._lazy_server_configs:
        _forget_lazy_server(key)


def _has_live_session(name: str, scope: Optional[str]) -> bool:
    from tools import mcp_tool as core
    from tools.mcp_tool_scope import _resolve_server_key

    with core._lock:
        server = core._servers.get(_resolve_server_key(name, scope, current=False))
    return server is not None and getattr(server, "session", None) is not None


def _visible_tools(name: str) -> List[str]:
    try:
        from tools.registry import registry
        return registry.get_tool_names_for_toolset(f"mcp-{name}")
    except Exception:
        return []


def _refresh_open_chats(name: str, row: Dict[str, Any]) -> None:
    """Hand this profile's open chats the new tool set with a one-line turn note, when this
    process hosts chats (``tui_gateway.server`` loaded)."""
    server = sys.modules.get("tui_gateway.server")
    if server is None:
        return
    if row["removed"]:
        note = f"[MCP server {name} removed]"
    elif row["connected"]:
        note = f"[MCP server {name} reloaded: {len(row['tools'])} tools]"
    else:
        note = f"[MCP server {name} not connected: {row.get('error') or 'unknown error'}]"
    try:
        from pathlib import Path
        from hermes_constants import get_hermes_home
        server.refresh_plugin_sessions(Path(get_hermes_home()), note)
    except Exception:
        logger.warning("open chats were not refreshed after reloading MCP server %r", name, exc_info=True)
