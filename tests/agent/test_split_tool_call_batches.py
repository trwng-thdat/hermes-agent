"""A ``tool_call`` that batches local tools is expanded into one call per entry before the
assistant message is recorded, instead of costing an LLM round on the single-entry rejection."""

import json
from types import SimpleNamespace

from agent.tool_dispatch_helpers import split_local_tool_call_batches

TAGS = {"name": "mcp__helpdesk__post_tags", "arguments": {"tagName": "super-sad"}}
SEARCH = {"name": "mcp__helpdesk__search_customers", "arguments": {"tag_ids": ["tag_1"]}}
MAIL = {"name": "connectors__gmail__SEND_EMAIL", "arguments": {"to": "a@b.c"}}
DRAFT = {"name": "connectors__gmail__CREATE_DRAFT", "arguments": {}}


def _tc(args, tc_id="call_1", name="tool_call"):
    raw = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=tc_id, type="function", function=SimpleNamespace(name=name, arguments=raw))


def _calls(tc):
    return json.loads(tc.function.arguments)["calls"]


def test_local_batch_becomes_one_call_per_entry_with_distinct_ids():
    original = _tc({"calls": [TAGS, SEARCH]})
    other = _tc({"query": "x"}, "call_2", name="web_search")
    out = split_local_tool_call_batches([original, other])
    assert [tc.id for tc in out] == ["call_1", "call_1_s2", "call_2"]
    assert [_calls(tc) for tc in out[:2]] == [[TAGS], [SEARCH]]
    assert out[2] is other
    assert _calls(original) == [TAGS, SEARCH]  # the emitted call is copied, never mutated


def test_a_dozen_local_entries_are_split():
    """Trace 253cbea5: a 12-tag count went over the old cap of 8 and cost a round on the rejection."""
    entries = [{"name": "mcp__helpdesk__get_customer", "arguments": {"tagIDs": f"tag_{n}"}} for n in range(12)]
    out = split_local_tool_call_batches([_tc({"calls": entries})])
    assert [_calls(tc) for tc in out] == [[e] for e in entries]
    assert len({tc.id for tc in out}) == 12


def test_connector_entries_stay_batched_in_place():
    out = split_local_tool_call_batches([_tc({"calls": [TAGS, MAIL, DRAFT, SEARCH]})])
    assert [_calls(tc) for tc in out] == [[TAGS], [MAIL, DRAFT], [SEARCH]]


def test_envelope_nested_inside_arguments_is_lifted():
    """Trace fb4efba4: `calls` nested (as a JSON string) inside `arguments` looped 8 times."""
    out = split_local_tool_call_batches([_tc({"arguments": json.dumps({"calls": [TAGS]})})])
    assert len(out) == 1 and json.loads(out[0].function.arguments) == {"calls": [TAGS]}


def test_dict_tool_calls_are_split_too():
    tc = {"id": "c9", "type": "function",
          "function": {"name": "tool_call", "arguments": json.dumps({"calls": [TAGS, SEARCH]})}}
    out = split_local_tool_call_batches([tc])
    assert [t["id"] for t in out] == ["c9", "c9_s2"]
    assert json.loads(out[1]["function"]["arguments"]) == {"calls": [SEARCH]}


def test_calls_that_need_no_split_return_the_same_list():
    calls = [
        _tc({"calls": [TAGS]}),
        _tc({"calls": [MAIL, DRAFT]}, "call_2"),
        _tc("not json {", "call_3"),
        _tc({"calls": [TAGS] * 65}, "call_4"),  # over the cap: the rejection still answers
        _tc({"calls": [TAGS, SEARCH]}, "call_5|fc_5"),  # Responses composite id
    ]
    assert split_local_tool_call_batches(calls) is calls
