"""``hooks.outbound`` must fire for a cron run whichever process executes it.

Regression: a dashboard "run now" (``POST /api/cron/jobs/{id}/trigger``) runs the job inside the
dashboard process, which never registered ``hooks.outbound`` — the run finished normally but no
``on_session_end`` was POSTed. Same for the detached ``python -m cron.scheduler
--external-worker-file`` worker. Only the gateway and CLI registered at startup.

End-to-end on the real path: dashboard route / worker entrypoint -> provider claim ->
``run_one_job`` -> ``run_job`` -> real ``AIAgent`` against an in-process OpenAI-compatible
stub -> turn finalizer ``on_session_end`` -> outbound webhook -> real local HTTP receiver.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class _LLMStub(BaseHTTPRequestHandler):
    """Answers every chat completion with one plain ``stop`` reply (streamed or not)."""

    def do_POST(self):  # noqa: N802 — http.server naming
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        if req.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"}, "finish_reason": None}]},
                {"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            ):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        body = json.dumps({
            "id": "m", "object": "chat.completion", "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class _Receiver(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 — http.server naming
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.server.captured.append({"path": self.path, "headers": dict(self.headers), "body": body})
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_args):
        pass


def _serve(handler):
    server = HTTPServer(("127.0.0.1", 0), handler)
    server.captured = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture()
def servers():
    llm, receiver = _serve(_LLMStub), _serve(_Receiver)
    yield llm, receiver
    for server in (llm, receiver):
        server.shutdown()
        server.server_close()


@pytest.fixture()
def profiles(tmp_path, monkeypatch, servers):
    """Default + ``worker_alpha`` homes, each with its OWN outbound target path and signing secret.

    Alpha's secret lives only in alpha's ``.env`` (``secret_env``); the launch process environ holds
    the DEFAULT profile's value under the same name, so a run that resolved it unscoped would sign
    with the wrong key."""
    from agent import outbound_webhooks
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.plugins import _reset_plugin_managers_for_tests

    llm, receiver = servers
    port = receiver.server_address[1]
    default_home = tmp_path / ".hermes"
    alpha_home = default_home / "profiles" / "worker_alpha"
    for name, home in (("default", default_home), ("worker_alpha", alpha_home)):
        (home / "cron").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            "model: test-model\n"
            "hooks:\n"
            "  outbound:\n"
            f"    - name: claw-{name}\n"
            f"      url: http://127.0.0.1:{port}/{name}\n"
            "      events: [on_session_end]\n"
            "      secret_env: CLAW_HOOK_SECRET\n",
            encoding="utf-8",
        )
        (home / ".env").write_text(f"CLAW_HOOK_SECRET=secret-{name}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("CLAW_HOOK_SECRET", "secret-default")
    monkeypatch.setattr(profiles_mod, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles_mod, "_get_profiles_root", lambda: default_home / "profiles")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kw: {
            "provider": "custom", "api_key": "test-key", "api_mode": "chat_completions",
            "base_url": f"http://127.0.0.1:{llm.server_address[1]}/v1",
        },
    )
    _reset_plugin_managers_for_tests()
    outbound_webhooks.reset_for_tests()
    yield {"default": default_home, "worker_alpha": alpha_home}
    outbound_webhooks.reset_for_tests()
    _reset_plugin_managers_for_tests()


@pytest.fixture()
def multiplexed():
    from agent.secret_scope import set_multiplex_active

    set_multiplex_active(True)
    yield
    set_multiplex_active(False)


def _session_end_deliveries(receiver, path):
    from agent import outbound_webhooks

    assert outbound_webhooks.flush(timeout=10.0)
    return [
        d for d in receiver.captured
        if d["path"] == path and json.loads(d["body"])["hook_event_name"] == "on_session_end"
    ]


def _assert_signed_cron_session_end(delivery, *, job_id, profile, secret):
    body = delivery["body"]
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert delivery["headers"]["X-Hermes-Signature-256"] == expected
    payload = json.loads(body)
    assert payload["profile"] == profile
    assert re.fullmatch(rf"cron_{job_id}_\d{{8}}_\d{{6}}", payload["session_id"])
    assert payload["extra"]["platform"] == "cron"
    assert payload["extra"]["completed"] is True
    assert payload["delivery_id"] == delivery["headers"]["X-Hermes-Delivery"]


@pytest.mark.asyncio
async def test_dashboard_trigger_delivers_the_owning_profiles_outbound_webhook(profiles, servers, multiplexed):
    """Dashboard run-now of a SECONDARY profile's job, in a multi-profile host: exactly one signed
    ``on_session_end`` reaches alpha's target, signed with alpha's secret; default's target gets none."""
    import hermes_cli.web_routers.cron as rt_cron
    import hermes_cli.web_server_cron as web_server_cron

    _llm, receiver = servers
    job = web_server_cron._call_cron_for_profile(
        "worker_alpha", "create_job", prompt="say done", schedule="every 1h", name="claw-run",
    )

    triggered = await rt_cron.trigger_cron_job(job["id"], profile="worker_alpha")

    assert triggered["last_status"] == "ok"
    deliveries = _session_end_deliveries(receiver, "/worker_alpha")
    assert len(deliveries) == 1
    _assert_signed_cron_session_end(
        deliveries[0], job_id=job["id"], profile="worker_alpha", secret="secret-worker_alpha")
    assert _session_end_deliveries(receiver, "/default") == []


def test_external_worker_delivers_outbound_webhook(profiles, servers, multiplexed):
    """The detached worker (``python -m cron.scheduler --external-worker-file``) runs the job in a
    fresh process that registered nothing at startup; its run still delivers to the job's profile."""
    import cron.scheduler as scheduler
    from cron.executions import mark_execution_handoff_pending
    from cron.jobs import create_job, use_cron_store
    from cron.scheduler_provider import InProcessCronScheduler
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _llm, receiver = servers
    alpha_home = profiles["worker_alpha"]
    token = set_hermes_home_override(str(alpha_home))
    try:
        with use_cron_store(alpha_home):
            job = create_job(prompt="say done", schedule="every 1h", name="claw-worker-run")
            claimed = InProcessCronScheduler().claim_fire(job["id"], manual=True)
            assert mark_execution_handoff_pending(claimed["execution_id"]) is not None
    finally:
        reset_hermes_home_override(token)
    handoff = alpha_home / "cron" / "external-workers"
    handoff.mkdir(parents=True, exist_ok=True)
    payload = handoff / f"{claimed['execution_id']}.json"
    payload.write_text(
        json.dumps({"job": claimed, "profile_home": str(alpha_home), "multiplex_active": True}),
        encoding="utf-8",
    )

    assert scheduler._run_external_worker_payload(payload, handoff / f"{claimed['execution_id']}.ready") is True

    deliveries = _session_end_deliveries(receiver, "/worker_alpha")
    assert len(deliveries) == 1
    _assert_signed_cron_session_end(
        deliveries[0], job_id=job["id"], profile="worker_alpha", secret="secret-worker_alpha")
    assert _session_end_deliveries(receiver, "/default") == []
