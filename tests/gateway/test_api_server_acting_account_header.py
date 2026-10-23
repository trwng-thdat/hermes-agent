"""X-Hermes-Acting-Account-Id (the chatting user's SSO account id) is validated like
X-Hermes-User-Id and bound as HERMES_SESSION_ACTING_ACCOUNT_ID for the API turn."""

from types import SimpleNamespace

import pytest

from gateway.platforms.api_server import ACTING_ACCOUNT_HEADER, APIServerAdapter
from gateway.session_context import clear_session_vars, get_session_env, reset_session_vars


def _adapter(api_key="k"):
    adapter = object.__new__(APIServerAdapter)
    adapter._api_key = api_key
    return adapter


def _request(**headers):
    return SimpleNamespace(headers={name: value for name, value in headers.items() if value is not None})


@pytest.mark.parametrize("value, api_key, status", [
    ("bad\r\nid", "k", 400), ("x" * 257, "k", 400), ("acc-42", "", 403)])
def test_header_rejected_like_the_user_id_header(value, api_key, status):
    parsed, err = _adapter(api_key)._parse_acting_account_header(_request(**{ACTING_ACCOUNT_HEADER: value}))
    assert parsed is None and err.status == status


@pytest.mark.parametrize("headers, expected", [
    ({ACTING_ACCOUNT_HEADER: " acc-42 "}, "acc-42"),
    ({"X-Hermes-User-Id": "u-1"}, None),
    ({}, None),
])
def test_header_is_parsed_and_trimmed(headers, expected):
    parsed, err = _adapter()._parse_acting_account_header(_request(**headers))
    assert err is None and parsed == expected


def test_header_is_bound_for_the_turn():
    parsed, err = _adapter()._parse_acting_account_header(_request(**{ACTING_ACCOUNT_HEADER: "acc-42"}))
    assert err is None
    reset_session_vars()
    tokens = APIServerAdapter._bind_api_server_session(user_id="u-1", acting_account_id=parsed)
    try:
        assert get_session_env("HERMES_SESSION_ACTING_ACCOUNT_ID") == "acc-42"
    finally:
        clear_session_vars(tokens)
        reset_session_vars()
