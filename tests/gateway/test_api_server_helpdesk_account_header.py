"""X-Hermes-Helpdesk-Account-Id is validated like X-Hermes-User-Id and bound as
HERMES_SESSION_HELPDESK_ACCOUNT_ID for the API turn."""

from types import SimpleNamespace

import pytest

from gateway.platforms.api_server import APIServerAdapter
from gateway.session_context import clear_session_vars, get_session_env, reset_session_vars

_HEADER = "X-Hermes-Helpdesk-Account-Id"


def _adapter(api_key="k"):
    adapter = object.__new__(APIServerAdapter)
    adapter._api_key = api_key
    return adapter


def _request(value):
    return SimpleNamespace(headers={_HEADER: value} if value is not None else {})


@pytest.mark.parametrize("value, api_key, status", [
    ("bad\r\nid", "k", 400), ("x" * 257, "k", 400), ("hd-42", "", 403)])
def test_header_rejected_like_the_user_id_header(value, api_key, status):
    parsed, err = _adapter(api_key)._parse_helpdesk_account_header(_request(value))
    assert parsed is None and err.status == status


def test_header_is_bound_for_the_turn():
    parsed, err = _adapter()._parse_helpdesk_account_header(_request(" hd-42 "))
    assert err is None
    reset_session_vars()
    tokens = APIServerAdapter._bind_api_server_session(user_id="u-1", helpdesk_account_id=parsed)
    try:
        assert get_session_env("HERMES_SESSION_HELPDESK_ACCOUNT_ID") == "hd-42"
    finally:
        clear_session_vars(tokens)
        reset_session_vars()
