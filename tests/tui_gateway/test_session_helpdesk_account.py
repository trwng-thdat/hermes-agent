"""``helpdesk_account_id`` on session.create / session.resume is bound as
HERMES_SESSION_HELPDESK_ACCOUNT_ID for each turn, alongside ``user_id``."""

import pytest

from gateway.session_context import get_session_env, reset_session_vars
from tui_gateway import server
from tui_gateway.contracts.registry import METHODS, validate_params
from tui_gateway.transport import bind_transport, reset_transport


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


@pytest.mark.parametrize("account_id", ["hd-42", ""])
def test_create_binds_helpdesk_account_for_the_turn_and_clears_after(created, account_id):
    session = created({"cols": 80, "user_id": "u-1", "helpdesk_account_id": account_id})
    tokens = server._set_session_context(session["session_key"])
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == "u-1"
        assert get_session_env("HERMES_SESSION_HELPDESK_ACCOUNT_ID", "unset") == account_id
    finally:
        server._clear_session_context(tokens)
    assert get_session_env("HERMES_SESSION_HELPDESK_ACCOUNT_ID", "unset") == ""


def test_resume_contract_accepts_user_and_helpdesk_account():
    params = {"session_id": "s", "user_id": "u-1", "helpdesk_account_id": "hd-42"}
    _, problem = validate_params(METHODS["session.resume"], params)
    assert problem is None
