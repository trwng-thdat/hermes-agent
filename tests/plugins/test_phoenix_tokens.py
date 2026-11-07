"""Phoenix plugin token counts: cache read/write per API call and per-turn totals."""

import importlib
from types import SimpleNamespace

phoenix = importlib.import_module("plugins.observability.phoenix")

PROMPT = phoenix._OI_LLM_TOKENS_PROMPT
COMPLETION = phoenix._OI_LLM_TOKENS_COMPLETION
TOTAL = phoenix._OI_LLM_TOKENS_TOTAL
CACHE_READ = phoenix._OI_LLM_TOKENS_CACHE_READ
CACHE_WRITE = phoenix._OI_LLM_TOKENS_CACHE_WRITE


def test_canonical_usage_counts_cache_inside_prompt():
    # post_api_request summary: input_tokens is the uncached part only.
    usage = {"input_tokens": 900, "output_tokens": 200, "cache_read_tokens": 16000,
             "cache_write_tokens": 100, "prompt_tokens": 17000, "total_tokens": 17200}
    assert phoenix._extract_tokens(None, usage) == {
        PROMPT: 17000, COMPLETION: 200, TOTAL: 17200, CACHE_READ: 16000, CACHE_WRITE: 100,
    }


def test_canonical_usage_without_prompt_total_rebuilds_it():
    usage = {"input_tokens": 900, "output_tokens": 200, "cache_read_tokens": 16000}
    tokens = phoenix._extract_tokens(None, usage)
    assert tokens[PROMPT] == 16900 and tokens[TOTAL] == 17100 and tokens[CACHE_WRITE] == 0


def test_canonical_usage_wins_over_raw_response():
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5))
    usage = {"input_tokens": 10, "output_tokens": 2, "cache_read_tokens": 30}
    assert phoenix._extract_tokens(response, usage)[PROMPT] == 40


def test_raw_openai_usage_keeps_cached_inside_prompt():
    raw = SimpleNamespace(prompt_tokens=17000, completion_tokens=200,
                          prompt_tokens_details=SimpleNamespace(cached_tokens=16000))
    tokens = phoenix._extract_tokens(SimpleNamespace(usage=raw), None)
    assert tokens[PROMPT] == 17000 and tokens[CACHE_READ] == 16000 and tokens[TOTAL] == 17200


def test_raw_anthropic_usage_adds_cache_beside_input():
    raw = {"input_tokens": 900, "output_tokens": 200, "cache_read_input_tokens": 16000,
           "cache_creation_input_tokens": 100}
    tokens = phoenix._extract_tokens(SimpleNamespace(usage=raw), None)
    assert tokens[PROMPT] == 17000 and tokens[CACHE_WRITE] == 100 and tokens[TOTAL] == 17200


def test_no_usage_gives_no_counts():
    assert phoenix._extract_tokens(None, None) == {}
    assert phoenix._extract_tokens(None, {"output_tokens": 0}) == {}


def test_turn_summary_totals_and_hit_rate():
    state = phoenix.TraceState(project="p", root_span=None, context=None)
    for usage in (
        {"input_tokens": 15000, "output_tokens": 300, "cache_read_tokens": 0},
        {"input_tokens": 500, "output_tokens": 200, "cache_read_tokens": 18000, "cache_write_tokens": 100},
    ):
        tokens = phoenix._extract_tokens(None, usage)
        state.total_tokens += tokens[TOTAL]
        state.prompt_tokens += tokens[PROMPT]
        state.completion_tokens += tokens[COMPLETION]
        state.cache_read_tokens += tokens[CACHE_READ]
        state.cache_write_tokens += tokens[CACHE_WRITE]
    assert phoenix._turn_token_summary(state) == {
        "total": 34100, "cache": 18100, "cache_read": 18000, "cache_write": 100,
        "input": 15500, "output": 500, "cache_hit_pct": 54,
    }


def test_turn_summary_empty_without_tokens():
    assert phoenix._turn_token_summary(phoenix.TraceState(project="p", root_span=None, context=None)) == {}
