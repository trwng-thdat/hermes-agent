"""``acting_account_id`` (the chatting user's SSO account id) on session.create / session.resume is
bound as HERMES_SESSION_ACTING_ACCOUNT_ID for each turn, alongside ``user_id``."""

import pytest

from gateway.session_context import get_session_env, reset_session_vars
from tui_gateway import server
from tui_gateway.contracts.registry import METHODS, validate_params
from tui_gateway.transport import bind_transport, reset_transport

_ENV = "HERMES_SESSION_ACTING_ACCOUNT_ID"


@pytest.fixture
def created(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_start_agent_build", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_completion_cwd", lambda params=None: str(tmp_path))
    token = bind_transport(None)
    sids = []

    def _create(params):
        resp = server.handle_request({"id": "r1", "method": "session.create", "params": params})
        assert "result" in resp, resp
        sids.append(resp["result"]["session_id"])
        return server._sessions[sids[-1]]

    yield _create
    reset_transport(token)
    for sid in sids:
        if (session := server._sessions.pop(sid, None)) is not None:
            server._teardown_session(session)
    reset_session_vars()


def _bound_account(session) -> str:
    tokens = server._set_session_context(session["session_key"])
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == "u-1"
        return get_session_env(_ENV, "unset")
    finally:
        server._clear_session_context(tokens)


@pytest.mark.parametrize("account_id", ["acc-42", ""])
def test_create_binds_acting_account_for_the_turn_and_clears_after(created, account_id):
    session = created({"cols": 80, "user_id": "u-1", "acting_account_id": account_id})
    assert _bound_account(session) == account_id
    assert get_session_env(_ENV, "unset") == ""


@pytest.mark.parametrize("method", ["session.create", "session.resume"])
def test_contract_accepts_the_acting_account(method):
    params = {"user_id": "u-1", "acting_account_id": "acc-42"}
    if method == "session.resume":
        params["session_id"] = "s"
    _, problem = validate_params(METHODS[method], params)
    assert problem is None


@pytest.mark.parametrize("params, expected", [
    ({"acting_account_id": "acc-42"}, "acc-42"),
    ({"acting_account_id": " acc-42 "}, "acc-42"),
    ({}, ""),
])
def test_resume_reads_the_acting_account(params, expected):
    from tui_gateway.methods_session import _Resume

    assert _Resume("r1", {"session_id": "s", **params}, "s").acting_account_id == expected
