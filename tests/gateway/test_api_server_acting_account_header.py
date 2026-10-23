"""X-Hermes-Acting-Account-Id (the chatting user's SSO account id) is validated like
X-Hermes-User-Id and bound as HERMES_SESSION_ACTING_ACCOUNT_ID for the API turn. The pre-rename
X-Hermes-Helpdesk-Account-Id is still read from older callers; the new header wins when both are sent."""

from types import SimpleNamespace

import pytest

from gateway.platforms.api_server import ACTING_ACCOUNT_HEADER, LEGACY_ACTING_ACCOUNT_HEADER, APIServerAdapter
from gateway.session_context import clear_session_vars, get_session_env, reset_session_vars


def _adapter(api_key="k"):
    adapter = object.__new__(APIServerAdapter)
    adapter._api_key = api_key
    return adapter


def _request(**headers):
    return SimpleNamespace(headers={name: value for name, value in headers.items() if value is not None})


@pytest.mark.parametrize("header", [ACTING_ACCOUNT_HEADER, LEGACY_ACTING_ACCOUNT_HEADER])
@pytest.mark.parametrize("value, api_key, status", [
    ("bad\r\nid", "k", 400), ("x" * 257, "k", 400), ("acc-42", "", 403)])
def test_header_rejected_like_the_user_id_header(header, value, api_key, status):
    parsed, err = _adapter(api_key)._parse_acting_account_header(_request(**{header: value}))
    assert parsed is None and err.status == status


@pytest.mark.parametrize("headers, expected", [
    ({ACTING_ACCOUNT_HEADER: " acc-42 "}, "acc-42"),
    ({LEGACY_ACTING_ACCOUNT_HEADER: "old-1"}, "old-1"),
    ({ACTING_ACCOUNT_HEADER: "new-1", LEGACY_ACTING_ACCOUNT_HEADER: "old-1"}, "new-1"),
    ({ACTING_ACCOUNT_HEADER: "new-1", LEGACY_ACTING_ACCOUNT_HEADER: "bad\r\nid"}, "new-1"),
    ({}, None),
])
def test_new_header_wins_over_the_legacy_one(headers, expected):
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
