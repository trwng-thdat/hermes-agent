"""``reload_mcp_server`` brings ONE server up to date for ONE profile without a restart (SCRUM-486).

Profiles A and B name server ``x`` with the same URL and token, so Hermes holds one live connection:
A owns it, B adopts it. A reload in one profile must not cut the other's tools (identity unchanged:
no teardown at all; identity changed: B is back by the time the call returns), must never touch
another server, and a disconnect in either order leaves ``x`` nowhere.

Real config files in temp profile homes, the real registration / adoption / shutdown code and a real
MCP event loop; only the network connect (``_run_discovery_pass``) and the transport task are fakes.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from hermes_constants import get_hermes_home, hermes_home_key, reset_hermes_home_override, set_hermes_home_override

_LEDGERS = ("_servers", "_server_scope_keys", "_server_tool_scopes", "_server_connecting",
            "_server_connect_errors", "_server_connect_retry_after", "_server_connect_failures",
            "_server_error_counts", "_server_breaker_opened_at", "_lazy_server_configs",
            "_mcp_tool_server_names", "_orphaned_adopters", "_parallel_safe_servers",
            "_server_trust_levels", "_tool_read_only_hints")


class _FakeServer:
    """A connected MCP server task: one tool, and a shutdown that drops its tools like the real one."""

    def __init__(self, name, cfg):
        self.name, self._config, self.session = name, cfg, object()
        self._tools = [SimpleNamespace(name="t", description="d", annotations=None,
                                       inputSchema={"type": "object", "properties": {}})]
        self.tool_timeout, self.initialize_result, self._sampling = 30, None, None
        self._registered_tool_names = []
        self.closed = False

    async def shutdown(self):
        from tools import mcp_tool_registration as reg
        for tool_name in list(self._registered_tool_names):
            reg._deregister_mcp_tool_all_scopes(self, tool_name)
        self._registered_tool_names, self.session, self.closed = [], None, True


def _cfg(token, **extra):
    return {"url": "https://mcp.example/x", "headers": {"Authorization": f"Bearer {token}"}, **extra}


@pytest.fixture
def hermes(tmp_path, monkeypatch):
    """Profile homes ``a``/``b`` (served, so scoped keys) plus the process's own home (``default``,
    bare keys); ``env.use(p)`` enters a home, ``env.write(p, servers)`` writes its config.yaml."""
    import tools.mcp_tool as core
    from tools import mcp_tool_config as _config, mcp_tool_discovery as disc, mcp_tool_lifecycle as lifecycle
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    homes = {"default": get_hermes_home()}
    for p in ("a", "b"):
        homes[p] = tmp_path / "profiles" / p
        homes[p].mkdir(parents=True)
    monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(_config, "_filter_suspicious_mcp_servers", lambda servers: servers)
    monkeypatch.setattr(disc._loop, "_ensure_mcp_loop", lambda: None)
    monkeypatch.setattr(lifecycle._loop, "_stop_mcp_loop", lambda **_kw: False)
    saved = {n: type(getattr(core, n))(getattr(core, n)) for n in _LEDGERS}
    for n in _LEDGERS:
        getattr(core, n).clear()

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(core, "_mcp_loop", loop)

    env = SimpleNamespace(homes=homes, shutdowns=[], down=set(), token=None, notes=[])
    # This process hosts chats: open chats of the reloaded profile get the new tools and a note.
    import tui_gateway.server as chat_server
    monkeypatch.setattr(chat_server, "refresh_plugin_sessions",
                        lambda home, note: env.notes.append((hermes_home_key(home), note)))

    def use(p):
        if env.token is not None:
            reset_hermes_home_override(env.token)
        env.token = set_hermes_home_override(homes[p]) if p != "default" else None
        return hermes_home_key(homes[p]) if p != "default" else None

    def write(p, servers):
        (homes[p] / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": servers}))

    def tools(p, server="x"):
        use(p)
        return registry.get_tool_names_for_toolset(f"mcp-{server}")

    def fake_pass(new_servers):
        for name, config in new_servers.items():
            if config["headers"]["Authorization"] in env.down:
                disc._note_connect_failure(name, ConnectionError("connection refused"))
                continue
            srv = _FakeServer(name, config)
            disc._adopt_server(name, srv)
            srv._registered_tool_names = reg._register_server_tools(name, srv, config)
            disc._note_connect_success(name)

    real_shutdown = lifecycle.shutdown_mcp_servers

    def spy_shutdown(**kwargs):
        env.shutdowns.append((kwargs.get("scope"), frozenset(kwargs.get("names") or ())))
        return real_shutdown(**kwargs)

    monkeypatch.setattr(lifecycle, "shutdown_mcp_servers", spy_shutdown)
    env.use, env.write, env.tools = use, write, tools
    with patch.object(disc, "_run_discovery_pass", fake_pass):
        yield env
    use("default")
    for tool_name in {e.name for e in registry.get_all_entries() if e.name.startswith("mcp__")}:
        for scope in [None] + [hermes_home_key(h) for h in homes.values()]:
            registry.deregister(tool_name, scope=scope)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()
    for n in _LEDGERS:
        getattr(core, n).clear()
        getattr(core, n).update(saved[n])


def _reload(env, profile, name="x"):
    from hermes_cli.mcp_reload_live import reload_mcp_server
    env.use(profile)
    return reload_mcp_server(name)


def _shared(env, extra=None):
    """A owns the live ``x`` connection (plus whatever ``extra`` servers), B adopted it."""
    import tools.mcp_tool as core
    servers = {"x": _cfg("shared"), **(extra or {})}
    env.write("a", servers)
    env.write("b", {"x": _cfg("shared")})
    _reload(env, "a")
    for other in extra or ():
        _reload(env, "a", other)
    _reload(env, "b")
    scope_a, scope_b = hermes_home_key(env.homes["a"]), hermes_home_key(env.homes["b"])
    assert list(core._servers) == [(scope_a, "x"), *[(scope_a, n) for n in extra or ()]]
    assert core._server_tool_scopes[(scope_a, "x")] == {scope_a, scope_b}
    assert env.tools("b") == ["mcp__x__t"]
    env.shutdowns.clear()
    env.notes.clear()
    return scope_a, scope_b


def test_unchanged_identity_reloads_without_teardown(hermes):
    import tools.mcp_tool as core
    scope_a, _ = _shared(hermes)
    owner = core._servers[(scope_a, "x")]

    for profile in ("a", "b"):
        row = _reload(hermes, profile)
        assert row == {"name": "x", "connected": True, "tools": ["mcp__x__t"], "removed": False,
                       "reconnected": False}

    assert hermes.shutdowns == [] and hermes.notes == []
    assert core._servers[(scope_a, "x")] is owner and not owner.closed
    assert hermes.tools("b") == ["mcp__x__t"]


def test_rotated_token_tears_down_only_the_owners_x_and_adopter_is_back(hermes):
    import tools.mcp_tool as core
    y_cfg = {"url": "https://mcp.example/y", "headers": {"Authorization": "Bearer y"}}
    scope_a, scope_b = _shared(hermes, {"y": y_cfg})
    old_x, y = core._servers[(scope_a, "x")], core._servers[(scope_a, "y")]
    hermes.write("a", {"x": _cfg("rotated"), "y": y_cfg})
    hermes.write("b", {"x": _cfg("rotated")})

    row = _reload(hermes, "a")

    assert row["connected"] and row["reconnected"] and row["tools"] == ["mcp__x__t"]
    assert hermes.shutdowns == [(scope_a, frozenset({"x"}))]
    assert hermes.notes == [(scope_a, "[MCP server x reloaded: 1 tools]")]
    assert old_x.closed and core._servers[(scope_a, "x")] is not old_x
    # B never reloaded, yet re-adopted A's new connection before A's call returned.
    assert core._server_tool_scopes[(scope_a, "x")] == {scope_a, scope_b}
    assert hermes.tools("b") == ["mcp__x__t"]
    # Y was never touched.
    assert core._servers[(scope_a, "y")] is y and not y.closed
    assert hermes.tools("a", "y") == ["mcp__y__t"]
    # B's own reload now finds the identity it wants: no teardown.
    hermes.shutdowns.clear()
    assert _reload(hermes, "b")["reconnected"] is False
    assert hermes.shutdowns == []


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
def test_disconnect_in_either_order_leaves_x_nowhere(hermes, order):
    import tools.mcp_tool as core
    y_cfg = {"url": "https://mcp.example/y", "headers": {"Authorization": "Bearer y"}}
    scope_a, _ = _shared(hermes, {"y": y_cfg})
    y = core._servers[(scope_a, "y")]
    hermes.write("a", {"x": _cfg("shared", enabled=False), "y": y_cfg})
    hermes.write("b", {})  # removed outright in B, disabled in A

    first = _reload(hermes, order[0])
    if order[0] == "b":
        # B only dropped its overlay: A's connection and tools are intact until A reloads.
        assert not core._servers[(scope_a, "x")].closed
        assert hermes.tools("b") == [] and hermes.tools("a") == ["mcp__x__t"]
    second = _reload(hermes, order[1])

    assert first["removed"] and second["removed"]
    assert not any(n == "x" for _, n in core._servers)
    assert hermes.tools("a") == [] and hermes.tools("b") == []
    assert core._servers[(scope_a, "y")] is y and not y.closed
    assert hermes.tools("a", "y") == ["mcp__y__t"]


def test_connect_failure_is_reported_not_raised(hermes):
    hermes.write("a", {"x": _cfg("dead")})
    hermes.down.add("Bearer dead")

    row = _reload(hermes, "a")

    assert row["connected"] is False and row["tools"] == [] and row["removed"] is False
    assert "connection refused" in row["error"]


def test_unreadable_config_is_reported_and_keeps_the_live_connection(hermes, monkeypatch):
    import tools.mcp_tool as core
    scope_a, _ = _shared(hermes)
    owner = core._servers[(scope_a, "x")]
    # x is still in a's config.yaml, but the interpolating loader failed and answered {}.
    monkeypatch.setattr("tools.mcp_tool_config._load_mcp_config", lambda: {})

    row = _reload(hermes, "a")

    assert row["removed"] is False and row["connected"] is False and "could not be read" in row["error"]
    assert hermes.shutdowns == [] and not owner.closed
    assert hermes.tools("b") == ["mcp__x__t"]


def test_reload_waits_for_the_chat_hosts_reload_mcp_lock(hermes):
    import tui_gateway.server as chat_server
    from hermes_cli.mcp_reload_live import reload_mcp_server
    hermes.write("a", {"x": _cfg("shared")})
    done = threading.Event()

    def reload_in_a():
        token = set_hermes_home_override(hermes.homes["a"])
        try:
            reload_mcp_server("x")
        finally:
            reset_hermes_home_override(token)
        done.set()

    with chat_server._mcp_reload_lock:
        worker = threading.Thread(target=reload_in_a)
        worker.start()
        assert not done.wait(0.3)
    assert done.wait(10)
    worker.join(timeout=5)


def test_default_profile_bare_key_never_touches_a_served_profiles_x(hermes):
    import tools.mcp_tool as core
    hermes.write("default", {"x": _cfg("default")})
    hermes.write("b", {"x": _cfg("b")})
    assert _reload(hermes, "default")["connected"]
    assert _reload(hermes, "b")["connected"]
    scope_b = hermes_home_key(hermes.homes["b"])
    b_x = core._servers[(scope_b, "x")]
    assert "x" in core._servers  # the default profile's own, bare-keyed connection

    hermes.write("default", {"x": _cfg("default", enabled=False)})
    assert _reload(hermes, "default")["removed"]

    assert "x" not in core._servers
    assert core._servers[(scope_b, "x")] is b_x and not b_x.closed
    assert hermes.tools("b") == ["mcp__x__t"]


def test_reload_route_runs_profiles_in_order_and_reports_failures_in_rows(monkeypatch):
    from starlette.testclient import TestClient

    import hermes_cli.mcp_reload_live as live
    from hermes_cli import web_server

    seen = []

    def fake_reload(name):
        seen.append((name, str(get_hermes_home())))
        if len(seen) == 1:
            raise RuntimeError("boom")  # reload_mcp_server never raises; the route must not 500 anyway
        return {"name": name, "connected": True, "tools": ["mcp__x__t"], "removed": False,
                "reconnected": False}

    monkeypatch.setattr(live, "reload_mcp_server", fake_reload)
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app)
    assert client.post("/api/mcp/servers/x/reload", json={"profiles": ["current"]}).status_code == 401

    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    response = client.post("/api/mcp/servers/x/reload",
                           json={"profiles": ["current", "no-such-profile", "Current"]})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True and list(body["results"]) == ["current", "no-such-profile", "Current"]
    assert body["results"]["current"]["connected"] is False and body["results"]["current"]["error"] == "boom"
    assert body["results"]["no-such-profile"]["connected"] is False
    assert "does not exist" in body["results"]["no-such-profile"]["error"]
    assert body["results"]["Current"]["connected"] is True
    assert [name for name, _ in seen] == ["x", "x"]
