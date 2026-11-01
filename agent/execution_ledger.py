"""Execution ledger: a deterministic receipt for every write tool call in a turn.

Each write result is classified in code (no LLM) as ``SUCCEEDED``, ``FAILED`` or ``UNKNOWN`` and
the receipt is merged into the result the model reads (``_receipt``), so the agent sees the real
outcome before it writes its answer. The ledger lives on the agent per turn and is the evidence the
chat claim gate (``agent.claim_stop``) checks before an answer may report an action as done.

- HTTP 2xx is not success by itself: a body that says ``success: false`` / ``status: FAILED`` /
  ``statusCode >= 400`` fails, and a response that echoes a different value than requested fails
  its postcondition. An authoritative response is accepted without a read-back.
- A timeout, a 5xx or a mid-call transport loss on a write is ``UNKNOWN``, never ``FAILED``: the
  request may have landed. The same call is refused until a read tool has checked the state.
- Re-running a write that already SUCCEEDED this turn with the same arguments is not executed;
  the earlier receipt comes back instead.

Wired in ``agent.tool_executor._dispatch_authorized_once``; every failure here fails open (the
tool result passes through unchanged)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

SUCCEEDED, FAILED, UNKNOWN = "SUCCEEDED", "FAILED", "UNKNOWN"
STATUSES = frozenset({SUCCEEDED, FAILED, UNKNOWN})
RECEIPT_KEY = "_receipt"

# --- write tools ---------------------------------------------------------------------------

_HELPDESK_TOOL = re.compile(r"(?:^|_)helpdesk_", re.I)
_HELPDESK_WRITE_TOOL = re.compile(
    r"(?:^|_)helpdesk_+(?:post|put|patch|delete|upsert|update|send|create|move|assign|add|publish|rollback)_", re.I
)
_WRITE_VERB = re.compile(r"(?:^|_)(?:create|update|delete|remove|send|schedule|cancel|add)(?:_|$)", re.I)
_MCP_PREFIX = re.compile(r"^mcp__(?P<server>.+?)__(?P<tool>.+)$")


def normalize_tool_name(name: str) -> str:
    """``mcp__<server>__<tool>`` → ``<tool>`` (``helpdesk_<tool>`` for a helpdesk server such as
    ``helpdesk`` or ``helpdesk_all``) so a verb in the server name never reads as a write."""
    m = _MCP_PREFIX.match(name or "")
    if not m:
        return name or ""
    server, tool = m.group("server"), m.group("tool")
    return f"helpdesk_{tool}" if re.search(r"(?:^|[_-])helpdesk(?:$|[_-])", server, re.I) else tool


def is_write_tool(name: str) -> bool:
    tool = normalize_tool_name(name)
    if _HELPDESK_TOOL.search(tool):
        return bool(_HELPDESK_WRITE_TOOL.search(tool))
    return bool(_WRITE_VERB.search(tool))


# --- result classification -----------------------------------------------------------------

_FAILED_STATUS = frozenset({"failed", "failure", "error", "rejected", "denied"})
_PENDING_STATUS = frozenset({"pending", "queued", "processing", "accepted", "in_progress", "running"})
# A definitive refusal from the API ("400: <message>", Helpdesk's generated-tool error shape).
_CLIENT_ERROR = re.compile(r"^\s*4\d\d\s*:")
# Errors after which the write may have landed: timeouts, 5xx, transport loss mid-call.
_UNCERTAIN_ERROR = re.compile(
    r"timed? ?out|timeout|took too long|internal error|upstream error|\b5\d\d\b|bad gateway|"
    r"service unavailable|connection (?:reset|closed|aborted)|aborted|broken pipe|may (?:or may not )?have",
    re.I,
)
# A create/add refused because the target already exists: the desired state holds.
_ALREADY_EXISTS = re.compile(r"đã tồn tại|already (?:exists?|assigned|added)|\bEXISTED\b", re.I)
# Response keys that describe the call, not the record; never compared with request arguments.
_META_KEYS = frozenset({
    "status", "statuscode", "status_code", "code", "message", "success", "ok", "error", "errors",
    "id", "_id", "createdat", "updatedat", "created_at", "updated_at", "tenant_id", "tenantid", "user_id",
})
_MAX_COMPARED_CHARS = 120


def _loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip()[:1] in ("{", "["):
        return None
    try:
        return json.loads(value)
    except ValueError:
        return None


def response_body(data: Any) -> Optional[dict]:
    """The API body inside an MCP envelope: ``{"result": "<json>"}`` (text content) or
    ``structuredContent``; ``data`` itself when it is already a plain body."""
    if not isinstance(data, dict):
        return None
    for key in ("structuredContent", "result"):
        inner = _loads(data.get(key))
        if isinstance(inner, dict):
            return inner
    return data


def body_failure(body: Any) -> Optional[str]:
    """The failure message a 2xx body carries (``success: false``, ``statusCode >= 400``,
    ``status: FAILED``, a non-empty ``error`` / ``errors``), else None."""
    if not isinstance(body, dict):
        return None
    message = str(body.get("message") or body.get("error") or "").strip()
    if body.get("success") is False or body.get("ok") is False:
        return message or "the API reported success=false"
    for key in ("statusCode", "status_code", "code"):
        code = body.get(key)
        if isinstance(code, int) and not isinstance(code, bool) and code >= 400:
            return message or f"status {code}"
    status = body.get("status")
    if isinstance(status, int) and not isinstance(status, bool) and status >= 400:
        return message or f"status {status}"
    if isinstance(status, str) and status.strip().lower() in _FAILED_STATUS:
        return message or f"status {status}"
    error = body.get("error")
    if error and isinstance(error, (str, dict, list)):
        return message or "the API returned an error"
    errors = body.get("errors")
    if isinstance(errors, list) and errors:
        return message or "the API returned errors"
    return None


def body_pending(body: Any) -> bool:
    """True for an accepted-but-not-done response (202 / ``status: pending``)."""
    if not isinstance(body, dict):
        return False
    if any(body.get(k) == 202 for k in ("statusCode", "status_code", "status", "code")):
        return True
    status = body.get("status")
    return isinstance(status, str) and status.strip().lower() in _PENDING_STATUS


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _comparable(value: Any) -> bool:
    """Short scalars only: long text may come back sanitized or reformatted."""
    if isinstance(value, str):
        return len(value) <= _MAX_COMPARED_CHARS
    return isinstance(value, (int, float, bool))


def postcondition_mismatch(args: Optional[dict], body: Any) -> Optional[str]:
    """A field the call asked to set that the response shows with another value, else None.
    Compared only where the response echoes the same key (top level or under ``data``); a list
    must contain every requested item. Without an echo there is nothing to compare."""
    if not isinstance(args, dict) or not isinstance(body, dict):
        return None
    records = [body] + [body[k] for k in ("data", "result") if isinstance(body.get(k), dict)]
    for key, wanted in args.items():
        if key.lower() in _META_KEYS or wanted is None or isinstance(wanted, dict):
            continue
        if isinstance(wanted, list):
            wanted = [v for v in wanted if _comparable(v)]
            if not wanted:
                continue
        elif not _comparable(wanted):
            continue
        for record in records:
            if key not in record or record[key] is None:
                continue
            got = record[key]
            if isinstance(wanted, list):
                have = {_norm(v) for v in got} if isinstance(got, list) else {_norm(got)}
                missing = [v for v in wanted if not isinstance(v, (dict, list)) and _norm(v) not in have]
                if missing:
                    return f"{key}: requested {missing} not in the response"
            elif not isinstance(got, (dict, list)) and _norm(got) != _norm(wanted):
                return f"{key}: requested {wanted!r}, the response shows {got!r}"
            break
    return None


@dataclass
class Receipt:
    operation_id: str
    tool: str
    status: str
    verified: bool
    args_digest: str
    evidence: str = ""
    error: str = ""
    dedup_of: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = {k: v for k, v in asdict(self).items() if v not in ("", None)}
        out.pop("args_digest", None)
        return out


def args_digest(name: str, args: Any) -> str:
    try:
        canonical = json.dumps(args or {}, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        canonical = repr(args)
    return hashlib.sha256(f"{normalize_tool_name(name)}\n{canonical}".encode("utf-8")).hexdigest()[:16]


def classify(name: str, result: Any, args: Optional[dict] = None) -> tuple:
    """``(status, verified, evidence, error)`` for one write result, decided in code."""
    data = result if isinstance(result, dict) else _loads(result)
    if not isinstance(data, dict):
        text = str(result or "")
        if text.startswith("Error"):
            uncertain = bool(_UNCERTAIN_ERROR.search(text))
            return (UNKNOWN if uncertain else FAILED), False, "tool raised", text[:300]
        # Plain-text success from a native tool: it ran, but nothing in it can be checked.
        return SUCCEEDED, False, "tool returned without an error", ""
    error = data.get("error")
    if data.get("outcome_uncertain"):
        return UNKNOWN, False, "outcome uncertain", str(error or "")[:300]
    if error:
        text = str(error)
        if _CLIENT_ERROR.match(text):
            return FAILED, True, "the API refused the request", text[:300]
        if _UNCERTAIN_ERROR.search(text):
            return UNKNOWN, False, "no final response from the API", text[:300]
        return FAILED, False, "the tool returned an error", text[:300]
    body = response_body(data)
    reason = body_failure(body)
    if reason:
        return FAILED, True, "the API response reports a failure", reason[:300]
    if body_pending(body):
        return UNKNOWN, False, "accepted but not completed yet", ""
    mismatch = postcondition_mismatch(args, body)
    if mismatch:
        return FAILED, True, "postcondition not met", mismatch[:300]
    return SUCCEEDED, True, "authoritative API response", ""


# --- ledger --------------------------------------------------------------------------------

@dataclass
class ExecutionLedger:
    turn_id: str
    receipts: List[Receipt] = field(default_factory=list)
    # Call order of reads after an UNKNOWN, so a retry is allowed only once state was checked.
    _reads_since: Dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, receipt: Receipt) -> None:
        with self._lock:
            self.receipts.append(receipt)

    def note_read(self) -> None:
        with self._lock:
            for op in self._reads_since:
                self._reads_since[op] += 1

    def latest(self, digest: str) -> Optional[Receipt]:
        """The last real execution of this operation (refused duplicates are skipped)."""
        with self._lock:
            for receipt in reversed(self.receipts):
                if receipt.args_digest == digest and not receipt.dedup_of:
                    return receipt
        return None


def ledger_for(agent: Any) -> Optional[ExecutionLedger]:
    """This turn's ledger (a new one when the turn id changed)."""
    turn_id = str(getattr(agent, "_current_turn_id", "") or "")
    ledger = getattr(agent, "_execution_ledger", None)
    if not isinstance(ledger, ExecutionLedger) or ledger.turn_id != turn_id:
        ledger = ExecutionLedger(turn_id=turn_id)
        try:
            agent._execution_ledger = ledger
        except Exception:
            return None
    return ledger


def _with_receipt(result: Any, receipt: Receipt) -> Any:
    """Merge the receipt into a JSON-object result; other shapes get it appended as a line."""
    if isinstance(result, str):
        data = _loads(result)
        if isinstance(data, dict):
            data[RECEIPT_KEY] = receipt.to_dict()
            try:
                return json.dumps(data, ensure_ascii=False)
            except (TypeError, ValueError):
                return result
        return f"{result}\n\n{RECEIPT_KEY}: {json.dumps(receipt.to_dict(), ensure_ascii=False)}"
    return result


def before_execute(agent: Any, name: str, args: Any, call_id: str) -> Optional[str]:
    """A synthetic result when the call must not run again, else None.

    - same tool + arguments already SUCCEEDED this turn → the earlier receipt, not executed;
    - same tool + arguments UNKNOWN and no read since → refused until the state is checked."""
    if not is_write_tool(name):
        try:
            ledger = ledger_for(agent)
            if ledger is not None:
                ledger.note_read()
        except Exception:
            logger.debug("execution ledger read note failed", exc_info=True)
        return None
    try:
        ledger = ledger_for(agent)
        if ledger is None:
            return None
        digest = args_digest(name, args)
        prior = ledger.latest(digest)
        if prior is None:
            return None
        if prior.status == SUCCEEDED:
            receipt = Receipt(
                operation_id=call_id or f"op_{digest}", tool=name, status=SUCCEEDED, verified=prior.verified,
                args_digest=digest, evidence="already done earlier this turn; not executed again",
                dedup_of=prior.operation_id,
            )
            ledger.record(receipt)
            return json.dumps({
                "result": f"Not executed again: the same call already succeeded this turn ({prior.operation_id}).",
                RECEIPT_KEY: receipt.to_dict(),
            }, ensure_ascii=False)
        if prior.status == UNKNOWN and ledger._reads_since.get(prior.operation_id, 0) == 0:
            receipt = Receipt(
                operation_id=call_id or f"op_{digest}", tool=name, status=UNKNOWN, verified=False,
                args_digest=digest, evidence="retry refused until the current state is checked",
                dedup_of=prior.operation_id,
            )
            ledger.record(receipt)
            return json.dumps({
                "error": (
                    f"Not executed: the outcome of the same call ({prior.operation_id}) is UNKNOWN and it may "
                    "already have been applied. Check the current state with a read tool first; call it again "
                    "only if the state shows it did not happen."
                ),
                "outcome_uncertain": True,
                RECEIPT_KEY: receipt.to_dict(),
            }, ensure_ascii=False)
    except Exception:
        logger.debug("execution ledger dedup check failed", exc_info=True)
    return None


def after_execute(agent: Any, name: str, args: Any, call_id: str, result: Any) -> Any:
    """Classify a write result, record its receipt and hand the model the result with it."""
    if not is_write_tool(name):
        return result
    try:
        ledger = ledger_for(agent)
        if ledger is None:
            return result
        digest = args_digest(name, args)
        prior = ledger.latest(digest)
        status, verified, evidence, error = classify(name, result, args if isinstance(args, dict) else None)
        if status == FAILED and prior is not None and prior.status == UNKNOWN and _ALREADY_EXISTS.search(error):
            # The earlier UNKNOWN call landed: this refusal proves the desired state holds.
            status, verified, evidence, error = SUCCEEDED, True, f"already applied by {prior.operation_id}", ""
        receipt = Receipt(
            operation_id=call_id or f"op_{digest}", tool=name, status=status, verified=verified,
            args_digest=digest, evidence=evidence, error=error,
        )
        ledger.record(receipt)
        if status == UNKNOWN:
            with ledger._lock:
                ledger._reads_since[receipt.operation_id] = 0
        return _with_receipt(result, receipt)
    except Exception:
        logger.debug("execution ledger record failed", exc_info=True)
        return result


# --- transcript fallback -------------------------------------------------------------------

def receipt_from_result(name: str, call_id: str, text: str, args: Optional[dict] = None) -> Optional[Receipt]:
    """A receipt for a write row in the transcript: the runtime's own ``_receipt`` when present,
    else a fresh classification (rows from before the ledger existed, tests)."""
    if not is_write_tool(name):
        return None
    data = _loads(text)
    stamped = data.get(RECEIPT_KEY) if isinstance(data, dict) else None
    if isinstance(stamped, dict) and stamped.get("status") in STATUSES:
        return Receipt(
            operation_id=str(stamped.get("operation_id") or call_id), tool=name, status=stamped["status"],
            verified=bool(stamped.get("verified")), args_digest=args_digest(name, args),
            evidence=str(stamped.get("evidence") or ""), error=str(stamped.get("error") or ""),
            dedup_of=str(stamped.get("dedup_of") or ""),
        )
    status, verified, evidence, error = classify(name, text, args)
    return Receipt(
        operation_id=call_id, tool=name, status=status, verified=verified,
        args_digest=args_digest(name, args), evidence=evidence, error=error,
    )


def summarize(receipts: Sequence[Receipt]) -> Dict[str, List[Receipt]]:
    """Final state per operation (a later call of the same tool + arguments wins)."""
    final: Dict[str, Receipt] = {}
    for receipt in receipts:
        final[receipt.args_digest] = receipt
    out: Dict[str, List[Receipt]] = {SUCCEEDED: [], FAILED: [], UNKNOWN: []}
    for receipt in final.values():
        out[receipt.status].append(receipt)
    return out
