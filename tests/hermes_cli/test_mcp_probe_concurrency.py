"""Concurrent probes share the MCP loop: only the last one to finish may stop it."""

import asyncio
import threading
from unittest.mock import patch

import pytest

from hermes_cli import mcp_config
from tools import mcp_tool as mcp_mod
from tools import mcp_tool_discovery
from tools import mcp_tool_lifecycle
from tools import mcp_tool_loop


def _close(coro):
    coro.close()


def test_first_probe_to_finish_does_not_stop_the_loop_under_a_running_probe():
    release_slow = threading.Event()
    slow_started = threading.Event()
    stops = []

    def fake_run(coro, timeout=None):
        name = coro.cr_frame.f_locals.get("name") if coro.cr_frame else None
        _close(coro)
        if name == "slow":
            slow_started.set()
            assert release_slow.wait(5)

    def fake_stop():
        stops.append(mcp_tool_lifecycle._active_probes)
        return True

    with patch("tools.mcp_tool_loop._ensure_mcp_loop"), \
            patch("tools.mcp_tool_loop._run_on_mcp_loop", side_effect=fake_run), \
            patch("tools.mcp_tool_lifecycle._stop_mcp_loop_if_idle", side_effect=fake_stop):
        slow = threading.Thread(
            target=mcp_config._probe_single_server, args=("slow", {"url": "https://slow.example/mcp"}))
        slow.start()
        assert slow_started.wait(5)

        mcp_config._probe_single_server("fast", {"url": "https://fast.example/mcp"})
        assert stops == []

        release_slow.set()
        slow.join(5)

    assert stops == [0]
    assert mcp_tool_lifecycle._active_probes == 0


def test_a_lone_probe_still_stops_the_idle_loop():
    with patch("tools.mcp_tool_loop._ensure_mcp_loop"), \
            patch("tools.mcp_tool_loop._run_on_mcp_loop", side_effect=lambda coro, timeout=None: _close(coro)), \
            patch("tools.mcp_tool_lifecycle._stop_mcp_loop_if_idle") as stop:
        mcp_config._probe_single_server("only", {"url": "https://only.example/mcp"})

    stop.assert_called_once()
    assert mcp_tool_lifecycle._active_probes == 0


def test_bulk_and_single_probes_do_not_stop_the_loop_under_each_other():
    """``probe_mcp_server_tools`` (bulk) and ``_probe_single_server`` share one probe counter."""
    release_bulk = threading.Event()
    bulk_started = threading.Event()
    stops = []

    def fake_run(coro_or_factory, timeout=None):
        if callable(coro_or_factory):  # bulk probe passes the coroutine function
            bulk_started.set()
            assert release_bulk.wait(5)
            return None
        _close(coro_or_factory)

    def fake_stop():
        stops.append(mcp_tool_lifecycle._active_probes)
        return True

    with patch("tools.mcp_tool._ensure_mcp_sdk", return_value=True), \
            patch.object(mcp_tool_discovery._config, "_load_mcp_config",
                         return_value={"bulk": {"url": "https://bulk.example/mcp"}}), \
            patch("tools.mcp_tool_loop._ensure_mcp_loop"), \
            patch("tools.mcp_tool_loop._run_on_mcp_loop", side_effect=fake_run), \
            patch("tools.mcp_tool_lifecycle._stop_mcp_loop_if_idle", side_effect=fake_stop):
        bulk = threading.Thread(target=mcp_tool_discovery.probe_mcp_server_tools)
        bulk.start()
        assert bulk_started.wait(5)

        # The single probe finishes while the bulk probe is still running: the loop must survive.
        mcp_config._probe_single_server("single", {"url": "https://single.example/mcp"})
        assert stops == []

        release_bulk.set()
        bulk.join(5)

    assert stops == [0]
    assert mcp_tool_lifecycle._active_probes == 0



@pytest.fixture
def idle_mcp_loop():
    """No registered servers and no loop running; whatever a test starts is stopped afterwards."""
    with mcp_mod._lock:
        mcp_mod._servers.clear()
        mcp_mod._server_connecting.clear()
    mcp_tool_loop._stop_mcp_loop()
    yield
    mcp_tool_loop._stop_mcp_loop()
    assert mcp_tool_lifecycle._active_probes == 0


def _runs_coroutines(loop) -> bool:
    async def _ping():
        return "pong"
    return asyncio.run_coroutine_threadsafe(_ping(), loop).result(timeout=5) == "pong"


def test_probe_starting_during_the_last_probes_stop_runs_on_a_live_loop(idle_mcp_loop):
    """The last probe out stops the loop under the probe lock; a probe entering meanwhile waits for
    the stop to finish, then gets a fresh running loop (never the one being torn down)."""
    real_stop = mcp_tool_loop._stop_mcp_loop
    stop_started, second_entered = threading.Event(), threading.Event()
    events, second = [], {}

    def slow_stop(**kwargs):
        if not stop_started.is_set():  # the first probe's stop; the second probe stops its own loop later
            stop_started.set()
            # The second probe is now blocked on entry; it must not get in while the stop runs.
            events.append("entered-during-stop" if second_entered.wait(0.3) else "stop-in-progress")
            result = real_stop(**kwargs)
            events.append("stopped")
            return result
        return real_stop(**kwargs)

    def second_probe():
        assert stop_started.wait(5)
        with mcp_tool_lifecycle._probe_mcp_loop():
            events.append("second-entered")
            second_entered.set()
            loop = mcp_mod._mcp_loop
            second["loop"] = loop
            second["works"] = loop is not None and loop.is_running() and _runs_coroutines(loop)

    thread = threading.Thread(target=second_probe)
    thread.start()
    with patch.object(mcp_tool_loop, "_stop_mcp_loop", side_effect=slow_stop):
        with mcp_tool_lifecycle._probe_mcp_loop():
            first_loop = mcp_mod._mcp_loop
            assert first_loop.is_running()
        thread.join(10)

    assert not thread.is_alive()
    assert events == ["stop-in-progress", "stopped", "second-entered"]
    assert second["works"] is True
    assert second["loop"] is not first_loop
    assert not first_loop.is_running()


def test_idle_stop_leaves_the_loop_running_under_an_in_flight_probe(idle_mcp_loop):
    with mcp_tool_lifecycle._probe_mcp_loop():
        loop = mcp_mod._mcp_loop
        assert mcp_tool_lifecycle._stop_mcp_loop_if_idle() is False
        # A profile-scoped shutdown (``/reload-mcp``) is an idle stop too.
        mcp_tool_lifecycle.shutdown_mcp_servers(scope="some-profile")
        assert mcp_mod._mcp_loop is loop
        assert _runs_coroutines(loop)
    # The last probe out still stops the now-idle loop.
    assert mcp_mod._mcp_loop is None
    assert not loop.is_running()


def test_process_wide_shutdown_still_stops_the_loop_under_a_probe(idle_mcp_loop):
    with mcp_tool_lifecycle._probe_mcp_loop():
        loop = mcp_mod._mcp_loop
        mcp_tool_lifecycle.shutdown_mcp_servers()
        assert mcp_mod._mcp_loop is None
        assert not loop.is_running()
    assert mcp_tool_lifecycle._active_probes == 0
