"""Text-response stop gates for the conversation turn loop.

When the model stops with a text answer, four gates may instead append the answer as an
interim row plus a synthetic user-role nudge and continue the turn: verify-on-stop (#65919),
the ``pre_verify`` plugin hook after code edits, the kanban worker terminal-tool guard, and the
chat claim gate (``agent.claim_stop``), which retracts an unsupported answer instead.
The first three keep the candidate answer as a budget-exhaustion fallback
(``pending_verification_response``) and clears ``final_response`` so the finalizer can tell
this gate from error exits (#61631). Nothing here imports ``agent.conversation_loop`` at
module level (cycle).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.message_metadata import append_message

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class StopGateVerdict:
    """``continue_turn`` True → a nudge was appended; re-enter the turn loop with
    ``final_response=None`` and the pending-verification fields updated."""

    continue_turn: bool
    final_response: Any
    pending_verification_response: Any
    pending_verification_response_previewed: Any


def _verify_on_stop_nudge(agent) -> Optional[str]:
    try:
        from agent.verification_stop import (
            build_verify_on_stop_nudge, verify_on_stop_enabled
        )

        if verify_on_stop_enabled():
            return build_verify_on_stop_nudge(
                session_id=getattr(agent, "session_id", None),
                changed_paths=getattr(agent, "_turn_file_mutation_paths", set()),
                attempts=getattr(agent, "_verification_stop_nudges", 0),
            )
    except Exception:
        logger.debug("verification stop-loop check failed", exc_info=True)
    return None


def _pre_verify_nudge(agent, final_response, attempt: int) -> Optional[str]:
    """After code edits a registered ``pre_verify`` hook may keep the agent going one
    more turn; no default continuation cost."""
    _edited = sorted(getattr(agent, "_turn_file_mutation_paths", set()) or [])
    try:
        from agent.verify_hooks import max_verify_nudges
        from hermes_cli.lifecycle import has_hook
        from hermes_cli.plugins import get_pre_verify_continue_message

        if _edited and has_hook("pre_verify") and attempt < max_verify_nudges():
            # Posture is fixed for the session — resolve once + cache.
            coding = getattr(agent, "_resolved_is_coding", None)
            if coding is None:
                from agent.coding_context import is_coding_context
                coding = bool(is_coding_context(platform=getattr(agent, "platform", "") or ""))
                agent._resolved_is_coding = coding
            return get_pre_verify_continue_message(
                session_id=getattr(agent, "session_id", None) or "",
                platform=getattr(agent, "platform", "") or "",
                model=getattr(agent, "model", "") or "", coding=coding, attempt=attempt,
                final_response=final_response, changed_paths=_edited,
            )
    except Exception:
        logger.debug("pre_verify hook check failed", exc_info=True)
    return None


def _kanban_stop_nudge(agent, messages) -> Optional[str]:
    """Workers must end with a terminal board tool (kanban_complete / kanban_block /
    kanban_request_review / kanban_request_changes); a narrated stop is recorded
    as protocol_violation, so nudge once or twice first."""
    try:
        from agent.kanban_stop import build_kanban_stop_nudge

        return build_kanban_stop_nudge(
            messages=messages, attempts=getattr(agent, "_kanban_stop_nudges", 0)
        )
    except Exception:
        logger.debug("kanban stop-loop check failed", exc_info=True)
        return None


def _claim_stop_nudge(
    agent, messages, final_response, current_turn_user_idx, api_call_count
) -> Optional[str]:
    """Chat answers must be backed by this turn's tool results (agent.claim_stop). Skipped once
    the cap is hit, on interrupt, and when the budget can't fund another call — so the budget
    fallback never meets a retracted answer."""
    try:
        from agent.claim_stop import (
            build_claim_nudge, check_answer, claim_gate_enabled, claim_judge_enabled, max_claim_nudges
        )

        if not final_response or not claim_gate_enabled(agent):
            return None
        attempts = getattr(agent, "_claim_stop_nudges", 0)
        if attempts >= max_claim_nudges() or getattr(agent, "_interrupt_requested", False):
            return None
        budget = getattr(agent, "iteration_budget", None)
        if (api_call_count or 0) + 1 >= agent.max_iterations or (
            budget is not None and budget.remaining <= 1
        ):
            return None
        if current_turn_user_idx is None:
            from agent.turn_context import reanchor_current_turn_user_idx
            current_turn_user_idx = reanchor_current_turn_user_idx(messages, None)
        hit = check_answer(
            agent, messages, final_response, current_turn_user_idx, judge=claim_judge_enabled()
        )
        if hit is None:
            return None
        kind, detail = hit
        logger.info("claim gate nudge issued (attempt %d) kind=%s detail=%s", attempts + 1, kind, detail)
        return build_claim_nudge(kind, detail)
    except Exception:
        logger.info("claim gate check failed open", exc_info=True)
        return None


def _append_interim_answer(agent, final_msg, messages, conversation_history, flush_fail_msg: str) -> None:
    """Real content: persist and emit as interim so the user sees the attempted answer;
    only the nudge is flagged synthetic (#65919)."""
    agent._emit_interim_assistant_message(final_msg)
    append_message(messages, final_msg)
    try:
        agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception:
        logger.debug(flush_fail_msg, exc_info=True)


def apply_stop_gates(
    agent: Any, final_msg: Dict[str, Any], *, final_response: Any, messages: List[Dict[str, Any]],
    conversation_history: Any, pending_verification_response: Any,
    pending_verification_response_previewed: Any, current_turn_user_idx: Any = None,
    api_call_count: Any = None,
) -> StopGateVerdict:
    """Run verify-on-stop → pre_verify hook → kanban stop guard → claim gate, in that order. Nudges
    are user-role rows appended only after the assistant answer row, so role alternation
    holds. Hook lookups are imported lazily from their origin modules (tests patch them
    there)."""

    def _continue(nudge: str, flag: str, *, fallback: bool = True) -> StopGateVerdict:
        """Append the synthetic nudge row and hand the turn back to the loop."""
        append_message(messages, {"role": "user", "content": nudge, flag: True})
        agent._session_messages = messages
        if not fallback:
            # A retracted answer must never come back as the budget fallback.
            return StopGateVerdict(
                continue_turn=True, final_response=None,
                pending_verification_response=pending_verification_response,
                pending_verification_response_previewed=pending_verification_response_previewed,
            )
        # Keep the answer only as a budget-exhaustion fallback; clear ``final_response`` so
        # the finalizer can tell this gate from error exits. Mark previewed only if the
        # candidate is reused (#61631).
        return StopGateVerdict(
            continue_turn=True, final_response=None,
            pending_verification_response=final_response,
            pending_verification_response_previewed=agent._interim_content_was_streamed(
                final_response or ""
            ),
        )

    _verify_nudge = _verify_on_stop_nudge(agent)
    if _verify_nudge:
        agent._verification_stop_nudges = getattr(agent, "_verification_stop_nudges", 0) + 1
        final_msg["finish_reason"] = "verification_required"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "verify-on-stop interim flush failed"
        )
        verdict = _continue(_verify_nudge, "_verification_stop_synthetic")
        # Internal nudge: stay silent on the terminal, debug-log only.
        logger.debug("verification stop-loop nudge issued (attempt %d)", agent._verification_stop_nudges)
        return verdict

    _attempt = getattr(agent, "_pre_verify_nudges", 0)
    _verify_nudge2 = _pre_verify_nudge(agent, final_response, _attempt)
    if _verify_nudge2:
        agent._pre_verify_nudges = _attempt + 1
        final_msg["finish_reason"] = "verify_hook_continue"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "pre_verify interim flush failed"
        )
        verdict = _continue(_verify_nudge2, "_pre_verify_synthetic")
        logger.debug("pre_verify nudge issued (attempt %d)", agent._pre_verify_nudges)
        return verdict

    _kanban_nudge = _kanban_stop_nudge(agent, messages)
    if _kanban_nudge:
        agent._kanban_stop_nudges = getattr(agent, "_kanban_stop_nudges", 0) + 1
        final_msg["finish_reason"] = "kanban_terminal_required"
        final_msg["_kanban_stop_synthetic"] = True
        append_message(messages, final_msg)
        verdict = _continue(_kanban_nudge, "_kanban_stop_synthetic")
        logger.info(
            "kanban stop-loop nudge issued (attempt %d) task=%s",
            agent._kanban_stop_nudges,
            os.environ.get("HERMES_KANBAN_TASK", ""),
        )
        agent._emit_diagnostic_status(
            "⚠️ Kanban worker tried to exit without a terminal board call "
            "(kanban_complete/kanban_request_review/kanban_block) — nudging to finish"
        )
        return verdict

    _claim_nudge = _claim_stop_nudge(agent, messages, final_response, current_turn_user_idx, api_call_count)
    if _claim_nudge:
        agent._claim_stop_nudges = getattr(agent, "_claim_stop_nudges", 0) + 1
        # Retracted: in context for the retry only — never persisted, emitted as interim or
        # reused as the fallback; the client drops the text it already streamed.
        final_msg["finish_reason"] = "claim_check_required"
        final_msg["_claim_rejected"] = True
        append_message(messages, final_msg)
        verdict = _continue(_claim_nudge, "_claim_stop_synthetic", fallback=False)
        try:
            agent._emit_status_kind("answer_retracted", "answer retracted", origin="claim_gate")
        except Exception:
            logger.debug("answer_retracted status emit failed", exc_info=True)
        return verdict
    return StopGateVerdict(
        continue_turn=False, final_response=final_response,
        pending_verification_response=pending_verification_response,
        pending_verification_response_previewed=pending_verification_response_previewed,
    )
