"""Chat answer claim gate: before a text answer ends the turn, an answer that reports an action as
done ("đã gửi", "successfully") must be backed by this turn's execution receipts
(``agent.execution_ledger``). Decided in code, no LLM call:

- an action turn (the user asked for something, or a write was attempted) with a completion claim
  and no SUCCEEDED write of a matching kind → retracted and nudged;
- a claim while a write is still FAILED / UNKNOWN that the answer never mentions → retracted;
- a question turn ("đã gửi chưa?") answered from history, or an answer with no claim → passes.

Once the nudge cap is hit the answer is replaced by a summary built from the receipts, so a false
claim never reaches the user. Every failure here fails open. Wired in ``agent.turn_stop_gates``."""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.execution_ledger import (  # noqa: F401  (is_write_tool is re-exported)
    FAILED, SUCCEEDED, UNKNOWN, Receipt, is_write_tool, normalize_tool_name, receipt_from_result, summarize,
)

logger = logging.getLogger("agent.conversation_loop")

_FALSY_TOKENS = {"0", "false", "no", "off"}
_TRUTHY_TOKENS = {"1", "true", "yes", "on"}

# ``auto`` scope: ai-claw's dashboard chat (backend session.create sends source "tool") and the
# local TUI/desktop. Cron, delegate children, kanban workers and messaging platforms stay off.
_AUTO_PLATFORMS = frozenset({"tui", "desktop", "tool"})
_DEFAULT_MAX_NUDGES = 2

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


def max_claim_nudges(config: Optional[dict] = None) -> int:
    try:
        return max(0, int(_agent_config(config).get("max_claim_nudges", _DEFAULT_MAX_NUDGES)))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_NUDGES


def hold_answer_stream(agent: Any) -> bool:
    """True once this turn ran a write under the claim gate: the answer is then withheld from the
    stream and delivered whole after the gate passes, so the user never sees it retracted."""
    ledger = getattr(agent, "_execution_ledger", None)
    turn_id = str(getattr(agent, "_current_turn_id", "") or "")
    if ledger is None or not ledger.receipts or ledger.turn_id != turn_id:
        return False
    cached = getattr(agent, "_claim_gate_hold", None)
    if not (isinstance(cached, tuple) and cached[0] == turn_id):
        try:
            cached = (turn_id, claim_gate_enabled(agent))
        except Exception:
            cached = (turn_id, False)
        agent._claim_gate_hold = cached
    return cached[1]


# --- completion claims ---------------------------------------------------------------------

_LETTER = r"[^\W\d_]"
# Claim kinds map an action verb to the write tools that can back it ("" = any write).
_VI_VERBS = {
    "send": r"gửi|nhắn|đăng",
    "create": r"tạo|thêm|lên lịch|đặt lịch|lưu",
    "update": r"cập nhật|đổi|chuyển|gắn|gán|phân công|bật|tắt|đóng|kích hoạt|thiết lập",
    "delete": r"xoá|xóa|hủy|huỷ|gỡ",
}
_EN_VERBS = {
    "send": r"sent|posted|published",
    "create": r"created|added|scheduled|saved",
    "update": r"updated|changed|moved|tagged|assigned|closed|enabled|disabled",
    "delete": r"deleted|removed|cancell?ed",
}
_KIND_TOOL_VERBS = {
    "send": re.compile(r"(?:^|_)(?:send|post|publish|message|messages|mail|email|broadcasts?|reply)(?:_|$)", re.I),
    "create": re.compile(r"(?:^|_)(?:create|post|add|upsert|schedule|insert|put)(?:_|$)", re.I),
    "update": re.compile(
        r"(?:^|_)(?:update|put|patch|post|assign|move|tags?|set|upsert|add|close|enable|disable)(?:_|$)", re.I
    ),
    "delete": re.compile(r"(?:^|_)(?:delete|remove|cancel)(?:_|$)", re.I),
}
_CLAIM_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = tuple(
    (kind, re.compile(rf"(?<!{_LETTER})đã\s+(?:được\s+)?(?:{verbs})(?!{_LETTER})", re.I))
    for kind, verbs in _VI_VERBS.items()
) + tuple(
    (kind, re.compile(
        rf"\b(?:have|has|was|were|been|i've|we've)\s+(?:(?!not\b|never\b)\w+\s+)?(?:{verbs})\b", re.I))
    for kind, verbs in _EN_VERBS.items()
) + (
    ("", re.compile(rf"(?<!{_LETTER})(?:hoàn tất|thành công|xong)(?!{_LETTER})", re.I)),
    ("", re.compile(r"\bsuccessfully\b|\b(?:all\s+)?done\b", re.I)),
)
# A negation anywhere earlier in the same clause makes it an honest admission, not a claim.
_NEGATION = re.compile(
    rf"(?<!{_LETTER})(?:chưa|không|chẳng)(?!{_LETTER})|\b(?:not|never|cannot|unable|failed|no)\b|\w+n't\b", re.I
)
# A conditional or future clause ("khi xong", "sẽ gửi", "once done") is not a report either.
_HYPOTHETICAL = re.compile(
    rf"(?<!{_LETTER})(?:khi|nếu|sẽ|để|sau khi|trước khi)(?!{_LETTER})|\b(?:when|once|after|if|will|to be)\b|\w+'ll\b", re.I
)
# ... as does a failure right after it ("đã gửi không thành công").
_FAILED_AFTER = re.compile(r"^\W*(?:\w+\s+){0,3}?(?:không thành công|thất bại|failed|unsuccessful)", re.I)
_CLAUSE_BREAK = re.compile(r"[,;:—–(|]|\s(?:nhưng|tuy nhiên|song|but|however)\s", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…\n])\s*")


def _clause_bounds(sentence: str, start: int, end: int) -> Tuple[str, str]:
    breaks_before = [m.end() for m in _CLAUSE_BREAK.finditer(sentence, 0, start)]
    after = _CLAUSE_BREAK.search(sentence, end)
    return sentence[breaks_before[-1] if breaks_before else 0:start], sentence[end:after.start() if after else None]


_PHRASE_TRIM = " \t*_`'\".,;:!?…)]|-–—"


def _quoted(sentence: str, start: int) -> bool:
    before = sentence[:start]
    return before.count('"') % 2 == 1 or before.count("“") > before.count("”")


def _cited(phrase: str, evidence: str) -> bool:
    """The claim phrase, through the end of its clause, is quoted from a tool result (a tag named
    "Đã tạo Tenant", a status "Gửi thành công"): data the answer repeats, not a report."""
    phrase = phrase.strip(_PHRASE_TRIM).lower()
    return bool(evidence) and " " in phrase and phrase in evidence


def _sentence_claim_kinds(sentence: str, evidence: str = "") -> List[str]:
    kinds = []
    for kind, pattern in _CLAIM_PATTERNS:
        for m in pattern.finditer(sentence):
            before, after = _clause_bounds(sentence, m.start(), m.end())
            if _NEGATION.search(before) or _HYPOTHETICAL.search(before) or _FAILED_AFTER.search(after):
                continue
            if _quoted(sentence, m.start()) or _cited(m.group() + after, evidence):
                continue
            kinds.append(kind)
            break
    return kinds


def find_claims(text: str, evidence: str = "") -> List[Tuple[str, str]]:
    """``[(sentence, kind)]`` for each non-question sentence claiming a completed action; kind is
    send / create / update / delete, or "" for a generic claim ("thành công", "done"). Phrases in
    quotes or quoted from *evidence* (this turn's tool results, lowercased) are not claims."""
    text = unicodedata.normalize("NFC", text or "")
    found = []
    for sentence in (s.strip() for s in _SENTENCE_SPLIT.split(text)):
        if sentence and not sentence.endswith("?"):
            for kind in _sentence_claim_kinds(sentence, evidence):
                found.append((sentence, kind))
    return found


def find_completion_claim(text: str) -> Optional[str]:
    """The first non-question sentence claiming a completed action, else None."""
    claims = find_claims(text)
    return claims[0][0] if claims else None


# Answers that tell the user something failed or is unconfirmed.
_ADMISSION = re.compile(
    rf"thất bại|không thành công|không thể|chưa (?:thể|được|xong|hoàn tất|xác nhận|rõ)|không rõ|(?<!{_LETTER})lỗi|"
    r"đang (?:xử lý|chờ)|\bfail(?:ed|ure)?\b|could(?: not|n't)|unable|\berror\b|not (?:yet )?(?:confirmed|"
    r"completed|done|sure)|unknown|pending",
    re.I,
)
# A user message that asks a question ("đã gửi chưa?") rather than for an action; a polite request
# phrased as a question ("gửi giúp mình được không?") still asks for the action.
_QUESTION_END = re.compile(rf"(?:\?|(?<!{_LETTER})(?:chưa|không|nào|gì|sao|hả|nhỉ|vậy)(?:\s*ạ)?)\s*$", re.I)
_QUESTION_WORD = re.compile(
    r"bao nhiêu|(?<!\w)mấy(?!\w)|^\s*(?:what|which|who|when|where|why|how|is|are|did|does|do|was|were|has|have)\b",
    re.I,
)
_POLITE_REQUEST = re.compile(r"giúp|(?<!\w)hộ(?!\w)|làm ơn|nhờ|được không|please|can you|could you|would you", re.I)


def is_question(text: str) -> bool:
    text = unicodedata.normalize("NFC", text or "").strip()
    if not text or _POLITE_REQUEST.search(text):
        return False
    return bool(_QUESTION_END.search(text) or _QUESTION_WORD.search(text))


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


def _call_parts(tc: Any) -> Tuple[str, str, Any]:
    if isinstance(tc, dict):
        fn = tc.get("function")
        name = fn.get("name") if isinstance(fn, dict) else tc.get("name")
        args = fn.get("arguments") if isinstance(fn, dict) else tc.get("arguments")
        return str(tc.get("id") or ""), str(name or ""), args
    fn = getattr(tc, "function", None)
    name = getattr(fn, "name", "") or getattr(tc, "name", "") or ""
    return str(getattr(tc, "id", "") or ""), str(name), getattr(fn, "arguments", None)


def _parse_args(raw: Any) -> Optional[dict]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            import json

            parsed = json.loads(raw or "{}")
            return parsed if isinstance(parsed, dict) else None
        except ValueError:
            return None
    return None


def _turn_rows(messages: Sequence[Any], start_idx: int) -> List[Tuple[str, str, str, Optional[dict]]]:
    """``[(call_id, tool_name, unwrapped_text, args)]`` for tool rows after this turn's user message."""
    calls: Dict[str, Tuple[str, Optional[dict]]] = {}
    rows = []
    for msg in list(messages)[max(start_idx, -1) + 1:]:
        if _skipped(msg):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or ():
                cid, name, args = _call_parts(tc)
                calls[cid] = (name, _parse_args(args))
        elif msg.get("role") == "tool":
            cid = str(msg.get("tool_call_id") or "")
            name, args = calls.get(cid, ("", None))
            name = name or str(msg.get("tool_name") or msg.get("name") or "")
            rows.append((cid, name, unwrap_tool_result(_text(msg.get("content"))), args))
    return rows


def turn_tool_results(messages: Sequence[Any], start_idx: int) -> List[ToolResult]:
    """``[(tool_name, ok, unwrapped_text)]`` for tool rows after this turn's user message."""
    from agent.display import _detect_tool_failure

    results: List[ToolResult] = []
    for _cid, name, text, _args in _turn_rows(messages, start_idx):
        try:
            failed = _detect_tool_failure(name, text)[0]
        except Exception:
            failed = False
        results.append((name, not failed, text))
    return results


def turn_receipts(agent: Any, messages: Sequence[Any], start_idx: int) -> List[Receipt]:
    """This turn's write receipts in call order: the live ledger's entry for a call when it has
    one, else the receipt stamped on (or classified from) the transcript row."""
    live: Dict[str, Receipt] = {}
    ledger = getattr(agent, "_execution_ledger", None)
    if ledger is not None and getattr(ledger, "turn_id", None) == str(getattr(agent, "_current_turn_id", "") or ""):
        live = {r.operation_id: r for r in list(ledger.receipts)}
    receipts = []
    for cid, name, text, args in _turn_rows(messages, start_idx):
        receipt = live.get(cid) or receipt_from_result(name, cid, text, args)
        if receipt is not None:
            receipts.append(receipt)
    return receipts


# --- decision ------------------------------------------------------------------------------

UNBACKED_CLAIM = "UNBACKED_CLAIM"
UNREPORTED_FAILURE = "UNREPORTED_FAILURE"


def _backs(receipt: Receipt, kind: str) -> bool:
    return not kind or bool(_KIND_TOOL_VERBS[kind].search(normalize_tool_name(receipt.tool)))


def _user_text(messages: Sequence[Any], start_idx: int) -> str:
    return "\n".join(
        _text(m.get("content")) for m in list(messages)[max(start_idx, 0):]
        if not _skipped(m) and m.get("role") == "user"
    )


def _json_strings(value: Any) -> List[str]:
    if isinstance(value, str):
        inner = _parse_json(value) if value.strip()[:1] in ("{", "[") else None
        return [value] + (_json_strings(inner) if inner is not None else [])
    if isinstance(value, dict):
        return [s for v in value.values() for s in _json_strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _json_strings(v)]
    return []


def _parse_json(text: str) -> Any:
    try:
        import json

        return json.loads(text)
    except ValueError:
        return None


def _tool_evidence(texts: Any) -> str:
    """Lowercased text of this turn's tool results, with JSON (also JSON nested in strings)
    decoded so values read as written: what an answer may quote without claiming anything."""
    parts = []
    for text in texts:
        parsed = _parse_json(text) if isinstance(text, str) else None
        parts.extend(_json_strings(parsed) if parsed is not None else [str(text or "")])
    return unicodedata.normalize("NFC", "\n".join(parts)).lower()


def check_answer(
    agent: Any, messages: Sequence[Any], answer: str, start_idx: int,
) -> Optional[Tuple[str, Any]]:
    """``(kind, claim_sentence)`` when the answer must not reach the user as written, else None."""
    rows = _turn_rows(messages, start_idx)
    if any(normalize_tool_name(name) in _SUBAGENT_TOOLS for _c, name, _t, _a in rows):
        return None  # the subagent's own writes are not in this transcript
    claims = find_claims(answer, _tool_evidence(text for _c, _n, text, _a in rows))
    if not claims:
        return None
    receipts = turn_receipts(agent, messages, start_idx)
    if not receipts and is_question(_user_text(messages, start_idx)):
        return None  # "đã gửi chưa?" answered from earlier turns: nothing to check here
    state = summarize(receipts)
    done = state[SUCCEEDED]
    for sentence, kind in claims:
        if not any(_backs(r, kind) for r in done):
            return UNBACKED_CLAIM, sentence
    if (state[FAILED] or state[UNKNOWN]) and not _ADMISSION.search(answer or ""):
        return UNREPORTED_FAILURE, claims[0][0]
    return None


# --- nudge and fallback --------------------------------------------------------------------

_NUDGE_LEAD = {
    UNBACKED_CLAIM: "Your last answer reports an action as done, but no tool call this turn succeeded at it",
    UNREPORTED_FAILURE: "Your last answer reports success, but some actions this turn failed or have an unknown outcome",
}

_VIETNAMESE = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ]")


def _reply_language(user_text: str) -> str:
    """The language the corrected answer must use; the nudge itself is English, and a small model
    otherwise answers it in English."""
    return "Vietnamese" if _VIETNAMESE.search(user_text or "") else "the user's language"


def _names(receipts: Sequence[Receipt], with_error: bool = False) -> str:
    out = []
    for r in receipts:
        label = f"{r.tool} ({r.operation_id})"
        out.append(f"{label}: {r.error}" if with_error and r.error else label)
    return ", ".join(dict.fromkeys(out))


def build_claim_nudge(kind: str, detail: Any, *, receipts: Sequence[Receipt] = (), user_text: str = "") -> str:
    state = summarize(receipts)
    lead = _NUDGE_LEAD.get(kind, _NUDGE_LEAD[UNBACKED_CLAIM])
    parts = [f"[System check] {lead}{': ' + repr(str(detail)) if detail else ''}."]
    if state[SUCCEEDED]:
        parts.append(f"SUCCEEDED this turn, do not call them again: {_names(state[SUCCEEDED])}.")
    if state[FAILED]:
        parts.append(
            "FAILED this turn; retry only if the error looks transient and the user's request still needs it, "
            f"otherwise tell the user it failed and why: {_names(state[FAILED], with_error=True)}."
        )
    if state[UNKNOWN]:
        parts.append(
            "UNKNOWN outcome (may already have been applied): check the current state with a read tool before "
            f"any retry, and never report these as done or failed until you know: {_names(state[UNKNOWN])}."
        )
    parts.append(
        "Call a write tool only for an action that has not succeeded this turn; otherwise state honestly what "
        "was and was not done. Keep everything in the previous answer that was correct. Write the complete "
        f"corrected answer in {_reply_language(user_text)}, addressed to the user. Do not open with a label such "
        "as \"Clarify:\" or \"Note:\", do not talk about tool calls, rounds or this check, and do not "
        "acknowledge, thank or apologise to it."
    )
    return " ".join(parts)


def _plain_tool(name: str) -> str:
    return re.sub(r"^helpdesk_", "", normalize_tool_name(name)).replace("_", " ")


def build_fallback_answer(receipts: Sequence[Receipt], user_text: str = "") -> str:
    """The answer that replaces an unsupported claim once the nudge cap is hit: only what the
    receipts prove, in the user's language."""
    state = summarize(receipts)
    vi = bool(_VIETNAMESE.search(user_text or ""))
    lines = [
        "Mình chưa thể xác nhận yêu cầu đã được thực hiện xong. Kết quả thực tế:" if vi else
        "I can't confirm the request was completed. What actually happened:"
    ]
    labels = (
        (SUCCEEDED, "Đã thực hiện" if vi else "Done"),
        (FAILED, "Thất bại" if vi else "Failed"),
        (UNKNOWN, "Chưa xác nhận được kết quả" if vi else "Outcome not confirmed"),
    )
    for status, label in labels:
        for r in state[status]:
            lines.append(f"- {label}: {_plain_tool(r.tool)}" + (f" ({r.error})" if r.error else ""))
    if len(lines) == 1:
        lines.append(
            "- Chưa có thao tác nào được thực hiện trong lượt này." if vi else
            "- No action was carried out this turn."
        )
    return "\n".join(lines)
