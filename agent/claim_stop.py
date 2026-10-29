"""Chat answer claim gate: before a text answer ends the turn, check it against this turn's
tool results. A completion claim ("đã gửi", "successfully") with no successful write tool, or a
concrete fact the grounding judge finds contradicted / unsourced, earns a nudge so the agent
corrects itself in the same turn. Pure helpers plus one auxiliary judge call; every failure
fails open (no nudge). Wired in ``agent.turn_stop_gates``."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("agent.conversation_loop")

_FALSY_TOKENS = {"0", "false", "no", "off"}
_TRUTHY_TOKENS = {"1", "true", "yes", "on"}

# ``auto`` scope: ai-claw's dashboard chat (backend session.create sends source "tool") and the
# local TUI/desktop. Cron, delegate children, kanban workers and messaging platforms stay off.
_AUTO_PLATFORMS = frozenset({"tui", "desktop", "tool"})
_DEFAULT_MAX_NUDGES = 2
_DEFAULT_JUDGE_TIMEOUT = 10.0

ToolResult = Tuple[str, bool, str]

# Rows the gate must not read as evidence or history: runtime scaffolding and rejected answers.
_SKIP_FLAGS = (
    "_claim_rejected", "_claim_stop_synthetic", "_verification_stop_synthetic", "_pre_verify_synthetic",
    "_kanban_stop_synthetic", "_dropped_toolcall_nudge", "_empty_recovery_synthetic",
    "_empty_terminal_sentinel", "_thinking_prefill", "_todo_snapshot_synthetic",
)
_SUBAGENT_TOOLS = frozenset({"subagent", "delegate_task"})


# --- enable resolution ---------------------------------------------------------------------

def _agent_config(config: Optional[dict]) -> dict:
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            config = {}
    agent_cfg = config.get("agent") if isinstance(config, dict) else None
    return agent_cfg if isinstance(agent_cfg, dict) else {}


def _env_flag(name: str) -> Optional[bool]:
    env = os.environ.get(name)
    return None if env is None else env.strip().lower() not in _FALSY_TOKENS


def _auto_scope(agent: Any) -> bool:
    if str(getattr(agent, "platform", "") or "").strip().lower() not in _AUTO_PLATFORMS:
        return False
    try:
        from agent.delegation_context import is_delegated_child_context, owned_kanban_task

        return not (is_delegated_child_context() or owned_kanban_task())
    except Exception:
        return True


def claim_gate_enabled(agent: Any, config: Optional[dict] = None) -> bool:
    """``HERMES_CLAIM_GATE`` env, then ``agent.claim_gate`` (bool or ``"auto"``, the default):
    ``auto`` is on only for the chat platforms in ``_AUTO_PLATFORMS``."""
    env = _env_flag("HERMES_CLAIM_GATE")
    if env is not None:
        return env
    val = _agent_config(config).get("claim_gate", "auto")
    if isinstance(val, bool):
        return val
    token = val.strip().lower() if isinstance(val, str) else "auto"
    if token in _TRUTHY_TOKENS:
        return True
    if token in _FALSY_TOKENS:
        return False
    return _auto_scope(agent)


def claim_judge_enabled(config: Optional[dict] = None) -> bool:
    env = _env_flag("HERMES_CLAIM_JUDGE")
    if env is not None:
        return env
    val = _agent_config(config).get("claim_judge", True)
    if isinstance(val, str):
        return val.strip().lower() not in _FALSY_TOKENS
    return bool(val)


def max_claim_nudges(config: Optional[dict] = None) -> int:
    try:
        return max(0, int(_agent_config(config).get("max_claim_nudges", _DEFAULT_MAX_NUDGES)))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_NUDGES


# --- completion-claim rule (port of apps/backend claim-verification.ts) --------------------

_HELPDESK_TOOL = re.compile(r"(?:^|_)helpdesk_", re.I)
_HELPDESK_WRITE_TOOL = re.compile(
    r"(?:^|_)helpdesk_+(?:post|put|patch|delete|upsert|update|send|create|move|assign|add|publish|rollback)_", re.I
)
_WRITE_VERB = re.compile(r"(?:^|_)(?:create|update|delete|remove|send|schedule|cancel|add)(?:_|$)", re.I)
_MCP_PREFIX = re.compile(r"^mcp__(?P<server>.+?)__(?P<tool>.+)$")

_LETTER = r"[^\W\d_]"
# Negated outcomes ("không thể tạo thành công", "couldn't be sent") are honest admissions, not
# claims. Up to three words may sit between the negation and the claim; a clause break ends it.
_NEGATION_GAP = r"(?:[^\s,;:.!?]+\s+){0,3}"
_VI_NEGATION_TAIL = re.compile(rf"(?<!{_LETTER})(?:chưa|không|chẳng)\s+{_NEGATION_GAP}$", re.I)
_EN_NEGATION_TAIL = re.compile(
    rf"(?:\b(?:not|never|cannot|unable\s+to|failed\s+to)|\w+n't)\s+{_NEGATION_GAP}$", re.I
)
_CLAIM_PATTERNS: Tuple[Tuple[re.Pattern, Optional[re.Pattern]], ...] = (
    (re.compile(rf"(?<!{_LETTER})đã\s+(?:gửi|tạo|cập nhật|xoá|xóa|lên lịch|đăng|thêm|lưu|hủy|huỷ)(?!{_LETTER})", re.I), None),
    (re.compile(rf"(?<!{_LETTER})hoàn tất(?!{_LETTER})", re.I), _VI_NEGATION_TAIL),
    (re.compile(rf"(?<!{_LETTER})thành công(?!{_LETTER})", re.I), _VI_NEGATION_TAIL),
    (re.compile(
        r"\b(?:have|has|was|were|been|i've|we've)\s+(?:(?!not\b|never\b)\w+\s+)?"
        r"(?:sent|created|updated|deleted|scheduled|saved|posted)\b", re.I), _EN_NEGATION_TAIL),
    (re.compile(r"\bsuccessfully\b", re.I), _EN_NEGATION_TAIL),
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…\n])\s*")


def _normalize_tool_name(name: str) -> str:
    """``mcp__<server>__<tool>`` → ``<tool>`` (``helpdesk_<tool>`` for a helpdesk server) so a
    verb in the server name never reads as a write."""
    m = _MCP_PREFIX.match(name or "")
    if not m:
        return name or ""
    server, tool = m.group("server"), m.group("tool")
    return f"helpdesk_{tool}" if re.search(r"(?:^|_)helpdesk$", server, re.I) else tool


def is_write_tool(name: str) -> bool:
    tool = _normalize_tool_name(name)
    if _HELPDESK_TOOL.search(tool):
        return bool(_HELPDESK_WRITE_TOOL.search(tool))
    return bool(_WRITE_VERB.search(tool))


def _sentence_claims(sentence: str) -> bool:
    for pattern, negation in _CLAIM_PATTERNS:
        for m in pattern.finditer(sentence):
            if negation is None or not negation.search(sentence[:m.start()]):
                return True
    return False


def find_completion_claim(text: str) -> Optional[str]:
    """The first non-question sentence claiming a completed action, else None."""
    text = unicodedata.normalize("NFC", text or "")
    for sentence in (s.strip() for s in _SENTENCE_SPLIT.split(text)):
        if sentence and not sentence.endswith("?") and _sentence_claims(sentence):
            return sentence
    return None


# --- this turn's tool results --------------------------------------------------------------

_UNTRUSTED_WRAPPER = re.compile(
    r"^\s*<untrusted_tool_result source=\"[^\"]*\">\n.*?\n\n(?P<body>.*)\n</untrusted_tool_result>\s*$", re.S
)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("text")
        )
    return "" if content is None else str(content)


def unwrap_tool_result(text: str) -> str:
    m = _UNTRUSTED_WRAPPER.match(text or "")
    return m.group("body") if m else (text or "")


def _skipped(msg: Any) -> bool:
    return not isinstance(msg, dict) or any(msg.get(flag) for flag in _SKIP_FLAGS)


def _call_id_and_name(tc: Any) -> Tuple[str, str]:
    if isinstance(tc, dict):
        fn = tc.get("function")
        name = fn.get("name") if isinstance(fn, dict) else tc.get("name")
        return str(tc.get("id") or ""), str(name or "")
    fn = getattr(tc, "function", None)
    return str(getattr(tc, "id", "") or ""), str(getattr(fn, "name", "") or getattr(tc, "name", "") or "")


def turn_tool_results(messages: Sequence[Any], start_idx: int) -> List[ToolResult]:
    """``[(tool_name, ok, unwrapped_text)]`` for tool rows after this turn's user message."""
    from agent.display import _detect_tool_failure

    names: Dict[str, str] = {}
    results: List[ToolResult] = []
    for msg in list(messages)[max(start_idx, -1) + 1:]:
        if _skipped(msg):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or ():
                cid, name = _call_id_and_name(tc)
                names[cid] = name
        elif msg.get("role") == "tool":
            name = names.get(str(msg.get("tool_call_id") or "")) or str(msg.get("tool_name") or msg.get("name") or "")
            text = unwrap_tool_result(_text(msg.get("content")))
            try:
                failed = _detect_tool_failure(name, text)[0]
            except Exception:
                failed = False
            results.append((name, not failed, text))
    return results


# --- grounding judge -----------------------------------------------------------------------

_VERDICTS = frozenset({"OK", "CONTRADICTED", "UNSOURCED", "UNBACKED_CLAIM"})
_TOOL_RESULT_MAX = 3_000
_TOOL_RESULTS_TOTAL = 12_000
_SYSTEM_PROMPT_MAX = 4_000
_HISTORY_MESSAGES = 10
_HISTORY_MESSAGE_MAX = 1_500

_JUDGE_INSTRUCTIONS = (
    "You fact-check an AI assistant's draft ANSWER before the user sees it. Compare it with the "
    "evidence: TOOL RESULTS from this turn, the USER MESSAGE, the RECENT CONVERSATION and the "
    "SYSTEM PROMPT.\n"
    "Check only concrete facts: numbers, counts, amounts, names, statuses, IDs, dates, times, links, "
    "emails, phone numbers, record contents, and statements that an action was performed.\n"
    "- CONTRADICTED: a concrete fact disagrees with a tool result.\n"
    "- UNBACKED_CLAIM: the answer says an action was done (sent, created, updated, deleted, scheduled...) "
    "but no tool result this turn shows it succeeded and no earlier turn shows it was done.\n"
    "- UNSOURCED: a concrete fact appears in none of the evidence.\n"
    "- OK: everything else.\n"
    "General knowledge, advice, explanations, plans, questions, phrasing, formatting, translation and "
    "reasonable rounding are fine. Saying it does not know or could not do something is fine. A fact "
    "backed by an earlier turn in the conversation is fine. When unsure, answer OK.\n"
    "Reply with JSON only: {\"verdict\": \"OK|CONTRADICTED|UNSOURCED|UNBACKED_CLAIM\", "
    "\"issues\": [{\"claim\": \"<short quote>\", \"why\": \"<short reason>\"}]} (issues empty when OK)."
)

# Digits (counts, prices, dates, IDs), links and emails: the concrete tokens worth a judge call.
_WORTHWHILE = re.compile(r"\d|https?://|www\.|[\w.+-]+@[\w-]+\.\w+")


def judge_worthwhile(answer: str, tool_results: Sequence[ToolResult]) -> bool:
    text = (answer or "").strip()
    if len(text) < 60 or text.endswith("?"):
        return False
    return bool(tool_results) or bool(_WORTHWHILE.search(text))


def parse_judge_reply(text: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    data: Any = None
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                data = json.loads(text[start:end + 1])
            except ValueError:
                return None
    if not isinstance(data, dict):
        return None
    verdict = str(data.get("verdict") or "").strip().upper()
    if verdict not in _VERDICTS:
        return None
    issues = [
        {"claim": str(i.get("claim") or ""), "why": str(i.get("why") or "")}
        for i in (data.get("issues") or []) if isinstance(i, dict)
    ]
    return {"verdict": verdict, "issues": issues}


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}… [clipped]"


def _format_tool_results(tool_results: Sequence[ToolResult]) -> str:
    out, budget = [], _TOOL_RESULTS_TOTAL
    for name, ok, text in tool_results:
        if budget <= 0:
            out.append("[more tool results omitted]")
            break
        body = _clip(text, min(_TOOL_RESULT_MAX, budget))
        budget -= len(body)
        out.append(f"[{name} — {'success' if ok else 'FAILED'}]\n{body}")
    return "\n\n".join(out) or "(no tools were called this turn)"


def _format_history(history: Sequence[Any]) -> str:
    rows = []
    for msg in history:
        text = _text(msg.get("content")).strip() if isinstance(msg, dict) else ""
        if text:
            role = msg.get("role")
            label = f"tool {msg.get('tool_name') or msg.get('name') or ''}".strip() if role == "tool" else role
            rows.append(f"{label}: {_clip(unwrap_tool_result(text), _HISTORY_MESSAGE_MAX)}")
    return "\n".join(rows) or "(none)"


def judge_grounding(
    agent: Any, *, user_text: str, answer: str, tool_results: Sequence[ToolResult],
    history: Sequence[Any], rule_hint: Optional[str],
) -> Optional[Dict[str, Any]]:
    """One auxiliary call; ``{"verdict", "issues"}`` or None on any failure (fail open)."""
    try:
        from agent.oneshot import run_oneshot

        try:
            from agent.auxiliary_client import _get_task_timeout

            timeout = _get_task_timeout("claim_gate", _DEFAULT_JUDGE_TIMEOUT)
        except Exception:
            timeout = _DEFAULT_JUDGE_TIMEOUT
        sections = [
            f"SYSTEM PROMPT:\n{_clip(str(getattr(agent, '_cached_system_prompt', '') or ''), _SYSTEM_PROMPT_MAX)}",
            f"RECENT CONVERSATION:\n{_format_history(history)}",
            f"USER MESSAGE:\n{user_text or '(none)'}",
            f"TOOL RESULTS:\n{_format_tool_results(tool_results)}",
        ]
        if rule_hint:
            sections.append(f"RULE CHECK HINT:\n{rule_hint}")
        sections.append(f"ANSWER:\n{answer}")
        user_input = "\n\n".join(sections)
        _note_judge_call(agent, judge_input=user_input, judge_started=time.time())
        reply = run_oneshot(
            instructions=_JUDGE_INSTRUCTIONS, user_input=user_input, task="claim_gate",
            max_tokens=300, temperature=0, timeout=timeout, main_runtime=agent._current_main_runtime(),
            reasoning_config={"enabled": False},  # a short JSON verdict needs no thinking
        )
        _note_judge_call(agent, judge_reply=reply, judge_ended=time.time())
    except Exception as exc:
        _note_judge_call(agent, judge_ended=time.time())
        logger.info("claim gate judge failed open: %s", exc)
        _note_judge_error(agent, str(exc))
        return None
    parsed = parse_judge_reply(reply)
    if parsed is None:
        logger.info("claim gate judge failed open: unparseable reply %r", (reply or "")[:200])
        _note_judge_error(agent, "unparseable reply")
    return parsed


def _note_judge_call(agent: Any, **fields: Any) -> None:
    """Record the judge's prompt, reply and timing on ``agent._claim_trace`` (observability)."""
    trace = getattr(agent, "_claim_trace", None)
    if isinstance(trace, dict):
        trace.update(fields)


def _note_judge_error(agent: Any, error: str) -> None:
    try:
        agent._claim_judge_error = error
    except Exception:
        pass


# --- nudge ---------------------------------------------------------------------------------

_NUDGE_LEAD = {
    "UNBACKED_CLAIM": "Your last answer reports an action as done, but no tool call this turn returned success for it",
    "CONTRADICTED": "Your last answer states facts that contradict this turn's tool results",
    "UNSOURCED": "Your last answer states concrete facts that come from no tool result, user message or instruction",
}


def turn_writes(tool_results: Sequence[ToolResult]) -> Tuple[List[str], List[str]]:
    """``(succeeded, failed)`` write tool names from this turn's results, in call order."""
    done = [n for n, ok, _ in tool_results if ok and is_write_tool(n)]
    failed = [n for n, ok, _ in tool_results if not ok and is_write_tool(n)]
    return done, failed


def build_claim_nudge(
    kind: str, detail: Any, *, done_writes: Sequence[str] = (), failed_writes: Sequence[str] = (),
) -> str:
    if isinstance(detail, (list, tuple)):
        named = "; ".join(
            f"\"{i.get('claim', '')}\" ({i.get('why', '')})" if i.get("why") else f"\"{i.get('claim', '')}\""
            for i in detail if isinstance(i, dict)
        )
    else:
        named = f"\"{detail}\"" if detail else ""
    lead = _NUDGE_LEAD.get(kind, _NUDGE_LEAD["UNSOURCED"])
    parts = [f"[System check] {lead}{': ' + named if named else ''}."]
    if done_writes:
        parts.append(f"Already done this turn, do not call them again: {', '.join(dict.fromkeys(done_writes))}.")
    if failed_writes:
        parts.append(
            f"Failed this turn, do not retry unless the user asks; tell the user it failed and why: "
            f"{', '.join(dict.fromkeys(failed_writes))}."
        )
    parts.append(
        "Call a write tool only for an action never attempted this turn; otherwise state honestly what "
        "was and was not done, or that you don't know. Write the complete corrected answer in the "
        "user's language. Reply to the user directly; do not acknowledge, thank or apologise to this check."
    )
    return " ".join(parts)


# --- decision ------------------------------------------------------------------------------

def check_answer(
    agent: Any, messages: Sequence[Any], answer: str, start_idx: int, *, judge: bool = True,
) -> Optional[Tuple[str, Any]]:
    """``(kind, detail)`` when the answer should be retracted, else None. With the judge on, a
    rule hit is only a hint and the judge decides; a judge failure falls back to the rule."""
    tool_results = turn_tool_results(messages, start_idx)
    rule_hit = None
    if not any(_normalize_tool_name(n) in _SUBAGENT_TOOLS for n, _, _ in tool_results):
        claim = find_completion_claim(answer)
        if claim and not any(ok and is_write_tool(n) for n, ok, _ in tool_results):
            rule_hit = claim
    verdict = None
    trace: Dict[str, Any] = {"rule_hit": rule_hit, "judge": "skipped" if judge else "off", "judge_error": ""}
    try:
        agent._claim_trace = trace  # read by the claim_gate observability hook
        agent._claim_judge_error = ""
    except Exception:
        pass
    if judge and (rule_hit or judge_worthwhile(answer, tool_results)):
        earlier = [m for m in list(messages)[:max(start_idx, 0)] if not _skipped(m)][-_HISTORY_MESSAGES:]
        user_text = "\n".join(
            _text(m.get("content")) for m in list(messages)[max(start_idx, 0):]
            if not _skipped(m) and m.get("role") == "user"
        )
        hint = None
        if rule_hit:
            writes = [f"{n}:{'ok' if ok else 'failed'}" for n, ok, _ in tool_results if is_write_tool(n)]
            hint = f"Completion claim \"{rule_hit}\"; write tools this turn: {', '.join(writes) or 'none'}."
        verdict = judge_grounding(
            agent, user_text=user_text, answer=answer, tool_results=tool_results,
            history=earlier, rule_hint=hint,
        )
        error = str(getattr(agent, "_claim_judge_error", "") or "")
        trace["judge"] = "ok" if verdict is not None else ("timeout" if "timed out" in error.lower() else "error")
        trace["judge_error"] = error
        if getattr(agent, "_interrupt_requested", False):
            return None
    if verdict is not None:
        return None if verdict["verdict"] == "OK" else (verdict["verdict"], verdict["issues"] or rule_hit)
    return ("UNBACKED_CLAIM", rule_hit) if rule_hit else None
