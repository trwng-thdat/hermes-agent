"""phoenix — Hermes plugin tracing agent turns, LLM calls and tool usage to Arize Phoenix.

Opt-in observability plugin. It mirrors the lifecycle hooks of the bundled
``observability/langfuse`` plugin but emits OpenTelemetry spans annotated with
OpenInference semantic conventions, exported over OTLP/HTTP to a Phoenix
collector.

The three product flows are routed to three separate Phoenix **projects** (the
``openinference.project.name`` resource attribute), decided per turn from the
``platform`` the turn runs under:

    platform == "cron"        -> project "cron"        (cron-scheduled turns)
    platform == "api_server"  -> project "wiki"        (agentic-ingest /v1/runs)
    everything else           -> project "agent-chat"  (interactive chat)

One TracerProvider is held per project so their spans land in distinct Phoenix
projects within a single process.

Hooks are inert (fail-open, never raise) when the OpenTelemetry SDK is missing
or ``HERMES_PHOENIX_ENDPOINT`` is unset. Env:

    HERMES_PHOENIX_ENDPOINT   (required) OTLP/HTTP traces endpoint,
                              e.g. http://localhost:6006/v1/traces
    HERMES_PHOENIX_HEADERS    optional  comma-separated k=v export headers
                              (Phoenix Cloud: "api_key=...")
    HERMES_PHOENIX_PROJECT_CHAT / _WIKI / _CRON   project-name overrides
    HERMES_PHOENIX_SERVICE_NAME   service.name resource attr (default hermes-agent)
    HERMES_PHOENIX_CAPTURE    metadata | sanitized (default) | full
    HERMES_PHOENIX_MAX_CHARS  max chars per captured field (default 12000)
    HERMES_PHOENIX_DEBUG      verbose plugin logging

See README.md.
"""
from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── OpenInference semantic-convention attribute keys ──────────────────────────
# Hard-coded so the plugin needs only the OTel SDK (hermes-agent[otlp]), not the
# separate openinference-semantic-conventions package.
_OI_SPAN_KIND = "openinference.span.kind"
_OI_INPUT_VALUE = "input.value"
_OI_INPUT_MIME = "input.mime_type"
_OI_OUTPUT_VALUE = "output.value"
_OI_OUTPUT_MIME = "output.mime_type"
_OI_LLM_MODEL = "llm.model_name"
_OI_LLM_PROVIDER = "llm.provider"
_OI_LLM_SYSTEM = "llm.system"
_OI_LLM_TOKENS_PROMPT = "llm.token_count.prompt"
_OI_LLM_TOKENS_COMPLETION = "llm.token_count.completion"
_OI_LLM_TOKENS_TOTAL = "llm.token_count.total"
_OI_TOOL_NAME = "tool.name"
_OI_SESSION_ID = "session.id"
_OI_USER_ID = "user.id"
_OI_METADATA = "metadata"
_OI_TAGS = "tag.tags"
_OI_PROJECT_RESOURCE = "openinference.project.name"

_MIME_JSON = "application/json"
_MIME_TEXT = "text/plain"

# Span-kind values (root turn is a CHAIN; each API call an LLM; tools TOOL; delegates AGENT).
_KIND_CHAIN = "CHAIN"
_KIND_LLM = "LLM"
_KIND_TOOL = "TOOL"
_KIND_AGENT = "AGENT"
_KIND_GUARDRAIL = "GUARDRAIL"

_TRACER_NAME = "hermes.observability.phoenix"


# ── lazy OTel SDK loading (fail-open) ─────────────────────────────────────────
_SDK: Optional[Dict[str, Any]] = None
_SDK_FAILED = object()
_SDK_LOCK = threading.Lock()


def _load_sdk() -> Optional[Dict[str, Any]]:
    """Import the OTel SDK symbols once; ``None`` (with a single warning) when the
    optional ``hermes-agent[otlp]`` extra is not installed."""
    global _SDK
    if _SDK is _SDK_FAILED:
        return None
    if _SDK is not None:
        return _SDK
    with _SDK_LOCK:
        if _SDK is _SDK_FAILED:
            return None
        if _SDK is not None:
            return _SDK
        try:
            from opentelemetry import trace as ot_trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.trace import set_span_in_context
            from opentelemetry.trace.status import Status, StatusCode
        except Exception as exc:  # pragma: no cover - fail-open when extra missing
            logger.warning(
                "Phoenix plugin is enabled but the OpenTelemetry SDK is unavailable; "
                "tracing is disabled. Install it with: pip install 'hermes-agent[otlp]' "
                "(import error: %s)",
                exc,
            )
            _SDK = _SDK_FAILED
            return None
        _SDK = {
            "trace": ot_trace,
            "OTLPSpanExporter": OTLPSpanExporter,
            "Resource": Resource,
            "TracerProvider": TracerProvider,
            "BatchSpanProcessor": BatchSpanProcessor,
            "set_span_in_context": set_span_in_context,
            "Status": Status,
            "StatusCode": StatusCode,
        }
        return _SDK


# ── config helpers ────────────────────────────────────────────────────────────
def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _debug(message: str) -> None:
    if _env("HERMES_PHOENIX_DEBUG").lower() in {"1", "true", "yes", "on"}:
        logger.info("Phoenix tracing: %s", message)


@contextlib.contextmanager
def _failsafe(label: str):
    """Swallow + debug-log any exception: telemetry must never block the agent turn."""
    try:
        yield
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"{label} failed: {exc}")


def _endpoint() -> str:
    return _env("HERMES_PHOENIX_ENDPOINT")


def _parse_headers() -> Dict[str, str]:
    """Parse ``k=v,k2=v2`` export headers (e.g. Phoenix Cloud ``api_key=...``)."""
    raw = _env("HERMES_PHOENIX_HEADERS")
    headers: Dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        key, value = key.strip(), value.strip()
        if key and value:
            headers[key] = value
    return headers


# ── capture modes (metadata | sanitized | full) ───────────────────────────────
_CAPTURE_MODES = ("metadata", "sanitized", "full")
_DEFAULT_CAPTURE_MODE = "sanitized"
_warned_invalid_capture = False


def _capture_mode() -> str:
    global _warned_invalid_capture
    value = _env("HERMES_PHOENIX_CAPTURE").lower()
    if not value or value in _CAPTURE_MODES:
        return value or _DEFAULT_CAPTURE_MODE
    if not _warned_invalid_capture:
        _warned_invalid_capture = True
        logger.warning(
            "Phoenix plugin: invalid HERMES_PHOENIX_CAPTURE=%r, falling back to %r (valid: %s)",
            value, _DEFAULT_CAPTURE_MODE, ", ".join(_CAPTURE_MODES),
        )
    return _DEFAULT_CAPTURE_MODE


def _max_chars() -> int:
    raw = _env("HERMES_PHOENIX_MAX_CHARS")
    if not raw:
        return 12000
    try:
        value = int(raw)
        return value if value > 0 else 12000
    except ValueError:
        return 12000


def _redact(value: str) -> str:
    # force=True: redact even if security.redact_secrets is off — this leaves the process.
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(value, force=True)
    except Exception:
        return value


def _capture_text(value: str) -> str:
    """Apply the active capture mode to a text value, returning a string."""
    mode = _capture_mode()
    if mode == "metadata":
        return f"[omitted: {len(value)} chars]"
    if mode == "sanitized":
        value = _redact(value)
    limit = _max_chars()
    over = len(value) - limit
    return value if over <= 0 else value[:limit] + f"... [truncated {over} chars]"


def _capture_json(value: Any) -> str:
    """Serialize CONTENT to a JSON string under the active capture mode."""
    if _capture_mode() == "metadata":
        return json.dumps(_describe(value))
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        text = str(value)
    return _capture_text(text)


def _describe(value: Any) -> Any:
    """Metadata-mode stand-in: shape and size, never payload."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return {"type": "number"}
    if isinstance(value, str):
        return {"omitted": True, "type": "text", "chars": len(value)}
    if isinstance(value, dict):
        return {"omitted": True, "type": "object", "keys": [str(k) for k in list(value.keys())[:20]]}
    if isinstance(value, (list, tuple, set)):
        return {"omitted": True, "type": "array", "items": len(value)}
    return {"omitted": True, "type": type(value).__name__}


# ── per-project TracerProvider registry ───────────────────────────────────────
_PROVIDERS: Dict[str, Any] = {}
_TRACERS: Dict[str, Any] = {}
_PROVIDER_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def _project_for(platform: str, session_id: str) -> str:
    """Map a turn's origin to one of the three Phoenix projects (env-overridable)."""
    p = (platform or "").strip().lower()
    if p == "cron" or (session_id or "").startswith("cron_"):
        return _env("HERMES_PHOENIX_PROJECT_CRON") or "cron"
    if p == "api_server":
        return _env("HERMES_PHOENIX_PROJECT_WIKI") or "wiki"
    return _env("HERMES_PHOENIX_PROJECT_CHAT") or "agent-chat"


def _tracer_for_project(project: str) -> Optional[Any]:
    """A cached tracer bound to ``project``'s own TracerProvider, or None when
    the SDK/endpoint is unavailable. The first build per project is serialized."""
    sdk = _load_sdk()
    if sdk is None:
        return None
    endpoint = _endpoint()
    if not endpoint:
        return None

    tracer = _TRACERS.get(project)
    if tracer is not None:
        return tracer
    with _PROVIDER_LOCK:
        tracer = _TRACERS.get(project)
        if tracer is not None:
            return tracer
        try:
            resource = sdk["Resource"].create({
                "service.name": _env("HERMES_PHOENIX_SERVICE_NAME") or "hermes-agent",
                _OI_PROJECT_RESOURCE: project,
            })
            provider = sdk["TracerProvider"](resource=resource)
            exporter = sdk["OTLPSpanExporter"](endpoint=endpoint, headers=_parse_headers() or None)
            provider.add_span_processor(sdk["BatchSpanProcessor"](exporter))
        except Exception as exc:  # pragma: no cover - fail-open
            logger.warning("Phoenix plugin: could not initialize provider for %r: %s", project, exc)
            return None
        _PROVIDERS[project] = provider
        tracer = provider.get_tracer(_TRACER_NAME)
        _TRACERS[project] = tracer
        _register_atexit_locked()
        _debug(f"initialized project {project!r} → {endpoint}")
        return tracer


def _register_atexit_locked() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_shutdown_all)
        _ATEXIT_REGISTERED = True


def _shutdown_all() -> None:
    """atexit: flush + shut down every provider (also ends any dangling roots)."""
    with _STATE_LOCK:
        states = list(_TRACE_STATE.values())
        _TRACE_STATE.clear()
    for state in states:
        with _failsafe("atexit finalize"):
            _end_children(state, include_subagents=True)
            _end_span(state.root_span)
    with _PROVIDER_LOCK:
        providers = list(_PROVIDERS.values())
    for provider in providers:
        with _failsafe("provider shutdown"):
            provider.shutdown()


# ── in-process trace state (one root span tree per agent turn) ────────────────
@dataclass
class TraceState:
    project: str
    root_span: Any
    context: Any  # OTel context with root span set (parent for children)
    generations: Dict[str, Any] = field(default_factory=dict)
    tools: Dict[str, Any] = field(default_factory=dict)
    pending_tools_by_name: Dict[str, list] = field(default_factory=dict)
    subagents: Dict[str, Any] = field(default_factory=dict)
    last_updated_at: float = field(default_factory=time.time)
    # Readability: steps are numbered, tools nest under the step that called them, and each
    # step's input shows only the messages added since the previous step.
    step_no: int = 0
    step_context: Any = None
    seen_messages: int = 0
    total_tokens: int = 0
    tool_count: int = 0
    tool_errors: int = 0
    started_at: float = field(default_factory=time.time)
    step_of: Dict[int, int] = field(default_factory=dict)


_STATE_LOCK = threading.Lock()
_TRACE_STATE: Dict[str, TraceState] = {}
# Bound the leak from turns that never reach _finish_trace (tool-only/interrupted
# final steps); evict least-recently-updated over the cap.
_MAX_TRACE_STATE = 256

# Pre-turn gate verdicts (e.g. the JEV gate) fire BEFORE the turn's root span
# exists, so stash them by session_id and attach a GUARDRAIL child when the root
# opens. Guarded by _STATE_LOCK. Bounded: a turn that never opens a root (failed
# before any LLM call) would otherwise leak one entry per turn.
_PENDING_GATES: Dict[str, dict] = {}
_MAX_PENDING_GATES = 256


def _trace_key(task_id: str, session_id: str, *, turn_id: str = "", api_request_id: str = "") -> str:
    """In-process trace scope key for one agent turn. ``turn_id`` wins over
    ``api_request_id`` so the turn-level post hook (no api_request_id) resolves to
    the same key as request-level hooks."""
    scope = f"task:{task_id}" if task_id else f"session:{session_id}" if session_id else f"thread:{threading.get_ident()}"
    if turn_id:
        return f"{scope}:turn:{turn_id}"
    if api_request_id:
        return f"{scope}:api:{api_request_id}"
    return task_id or scope


def _state_for_turn(turn_id: str) -> Optional[TraceState]:
    """Live state for a turn id alone (caller holds ``_STATE_LOCK``). Subagent hooks
    carry ``parent_turn_id`` but no ``task_id``; match on the ``:turn:<id>`` suffix."""
    if not turn_id:
        return None
    suffix = f":turn:{turn_id}"
    return next((state for key, state in _TRACE_STATE.items() if key.endswith(suffix)), None)


# ── span primitives ───────────────────────────────────────────────────────────
def _set_attr(span: Any, key: str, value: Any) -> None:
    if value is None:
        return
    with _failsafe(f"set_attribute {key}"):
        if isinstance(value, (str, bool, int, float)):
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, _capture_json(value))


def _set_metadata(span: Any, metadata: Optional[dict]) -> None:
    if not metadata:
        return
    clean = {k: v for k, v in metadata.items() if v is not None}
    if clean:
        with _failsafe("set metadata"):
            span.set_attribute(_OI_METADATA, json.dumps(clean, default=str, ensure_ascii=False))


def _end_span(span: Any) -> None:
    if span is None:
        return
    with _failsafe("span end"):
        span.end()


def _end_children(state: TraceState, *, include_subagents: bool = False) -> None:
    pending = [obs for queue in state.pending_tools_by_name.values() for obs in queue]
    subagents = state.subagents.values() if include_subagents else ()
    for span in (*state.generations.values(), *state.tools.values(), *pending, *subagents):
        _end_span(span)


def _start_root(task_key: str, *, project: str, tracer: Any, task_id: str, session_id: str,
                platform: str, provider: str, model: str, api_mode: str, messages: Any,
                turn_id: str, api_request_id: str) -> TraceState:
    sdk = _load_sdk()
    last_user = next((m for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"), None) \
        if isinstance(messages, list) else None
    user_text = _content_text(last_user.get("content")) if last_user is not None else ""
    span = tracer.start_span(_turn_name(user_text))
    _set_attr(span, _OI_SPAN_KIND, _KIND_CHAIN)
    if session_id:
        _set_attr(span, _OI_SESSION_ID, session_id)
    if last_user is not None:
        _set_attr(span, _OI_INPUT_VALUE, _capture_text(user_text))
        _set_attr(span, _OI_INPUT_MIME, _MIME_TEXT)
    _set_metadata(span, {
        "source": "hermes", "profile": _profile_name(), "task_id": task_id, "turn_id": turn_id,
        "api_request_id": api_request_id, "platform": platform, "provider": provider, "model": model,
        "api_mode": api_mode, "project": project, "capture_mode": _capture_mode(),
    })
    context = sdk["set_span_in_context"](span)
    state = TraceState(project=project, root_span=span, context=context)
    # Step 1's input starts at the user message: the system prompt and history are the same for
    # every step, so they show once, as step 1's system message, not on every step.
    if isinstance(messages, list) and last_user is not None:
        state.seen_messages = len(messages) - 1 - messages[::-1].index(last_user)
    # Attach any pre-turn gate verdict (JEV) stashed for this session as the
    # first child of the turn.
    pending = _PENDING_GATES.pop(session_id, None) if session_id else None
    if pending is not None:
        _emit_gate_span(state, pending)
    _debug(f"started root {task_key} in project {project!r}")
    return state


def _get_or_start_state_locked(task_key: str, **root_kwargs: Any) -> TraceState:
    """Caller holds ``_STATE_LOCK``. Start a root span tree if the key is new,
    first evicting least-recently-updated state down to the cap (evicted roots
    are ended so they don't dangle)."""
    state = _TRACE_STATE.get(task_key)
    if state is None:
        state = _start_root(task_key, **root_kwargs)
        over = len(_TRACE_STATE) - (_MAX_TRACE_STATE - 1)
        for key, stale in sorted(_TRACE_STATE.items(), key=lambda kv: kv[1].last_updated_at)[:max(over, 0)]:
            _TRACE_STATE.pop(key, None)
            _end_children(stale, include_subagents=True)
            _end_span(stale.root_span)
        _TRACE_STATE[task_key] = state
    state.last_updated_at = time.time()
    return state


def _start_child(state: TraceState, *, name: str, kind: str, context: Any = None,
                 start_time: Optional[int] = None) -> Any:
    tracer = _TRACERS.get(state.project)
    if tracer is None:
        return None
    span = tracer.start_span(name, context=context or state.context, start_time=start_time)
    _set_attr(span, _OI_SPAN_KIND, kind)
    return span


def _emit_gate_span(state: TraceState, payload: dict) -> None:
    """Emit a completed GUARDRAIL child span for a pre-turn tool gate (JEV)."""
    gate = str(payload.get("gate") or "gate")
    span = _start_child(state, name=f"{gate.upper()} gate", kind=_KIND_GUARDRAIL)
    if span is None:
        return
    with _failsafe("emit gate span"):
        _set_attr(span, _OI_INPUT_VALUE, _capture_text(str(payload.get("input_text") or "")))
        _set_attr(span, _OI_INPUT_MIME, _MIME_TEXT)
        _set_attr(span, _OI_OUTPUT_VALUE, _capture_json({"disable_tools": payload.get("disable_tools")}))
        _set_attr(span, _OI_OUTPUT_MIME, _MIME_JSON)
        _set_metadata(span, {
            "gate": gate,
            "decision": "disable_tools" if payload.get("disable_tools") else "keep_tools",
            "p_needs_tools": payload.get("p_needs_tools"),
            "skip_confidence": payload.get("skip_confidence"),
            "model": payload.get("model"),
            "status": payload.get("status"),
            "duration_ms": payload.get("duration_ms"),
        })
        _end_span(span)


def _request_key(api_call_count: Any) -> str:
    return str(api_call_count or 0)


def _finish_trace(task_key: str, *, output: Any = None, error: bool = False, status_message: str = "") -> None:
    with _STATE_LOCK:
        state = _TRACE_STATE.pop(task_key, None)
    if state is None:
        return
    with _failsafe("finish trace"):
        _end_children(state)
        if isinstance(output, str):
            _set_attr(state.root_span, _OI_OUTPUT_VALUE, _capture_text(output))
            _set_attr(state.root_span, _OI_OUTPUT_MIME, _MIME_TEXT)
        elif output is not None:
            _set_attr(state.root_span, _OI_OUTPUT_VALUE, _capture_json(output))
            _set_attr(state.root_span, _OI_OUTPUT_MIME, _MIME_JSON)
        if state.total_tokens:
            _set_attr(state.root_span, _OI_LLM_TOKENS_TOTAL, state.total_tokens)
        errors = f" ({state.tool_errors} lỗi)" if state.tool_errors else ""
        tokens = f" · {state.total_tokens} tok" if state.total_tokens else ""
        _set_attr(state.root_span, "hermes.summary",
                  f"{state.step_no} bước · {state.tool_count} tool{errors} · "
                  f"{round(time.time() - state.started_at, 1)}s{tokens}")
        if error:
            _mark_error(state.root_span, status_message)
        _end_span(state.root_span)


def _mark_error(span: Any, message: str) -> None:
    sdk = _load_sdk()
    if sdk is None or span is None:
        return
    with _failsafe("mark error"):
        span.set_status(sdk["Status"](sdk["StatusCode"].ERROR, (message or "error")[:200]))


# ── token usage extraction ────────────────────────────────────────────────────
def _extract_tokens(response: Any, usage: Any) -> Dict[str, int]:
    """(prompt, completion, total) tokens from a response object's ``.usage`` or a
    summary ``usage`` dict. Handles both input/output and prompt/completion naming."""
    src: Any = None
    if getattr(response, "usage", None) is not None:
        src = response.usage
    elif isinstance(usage, dict) and usage:
        src = usage

    def _get(*names: str) -> int:
        for n in names:
            v = src.get(n) if isinstance(src, dict) else getattr(src, n, None)
            if isinstance(v, (int, float)) and v:
                return int(v)
        return 0

    if src is None:
        return {}
    prompt = _get("input_tokens", "prompt_tokens")
    completion = _get("output_tokens", "completion_tokens")
    total = _get("total_tokens") or (prompt + completion)
    out: Dict[str, int] = {}
    if prompt:
        out[_OI_LLM_TOKENS_PROMPT] = prompt
    if completion:
        out[_OI_LLM_TOKENS_COMPLETION] = completion
    if total:
        out[_OI_LLM_TOKENS_TOTAL] = total
    return out


def _serialize_assistant(assistant_message: Any, assistant_response: Any,
                         assistant_content_chars: int, assistant_tool_call_count: int) -> Dict[str, Any]:
    """Best-effort assistant-output dict across the post_llm_call / post_api_request shapes."""
    if assistant_message is not None:
        content = getattr(assistant_message, "content", None)
        tool_calls = getattr(assistant_message, "tool_calls", None) or []
        reasoning = getattr(assistant_message, "reasoning", None) or getattr(assistant_message, "reasoning_content", None)
        return {"content": content, "reasoning": reasoning, "tool_calls": list(tool_calls)}
    if assistant_response is not None:
        return {"content": assistant_response, "reasoning": None, "tool_calls": []}
    return {
        "content": f"[{assistant_content_chars} chars]" if assistant_content_chars else None,
        "reasoning": None,
        "tool_calls": [{"id": f"tc_{i}"} for i in range(assistant_tool_call_count or 0)],
    }


# ── readable naming / content (what a reader of the Phoenix tree needs) ─────────
_MCP_NAME = re.compile(r"^mcp__(?P<server>.+?)__(?P<tool>.+)$")
_META_TOOLS = frozenset({"tool_call"})  # dispatchers whose real target sits in args["calls"]
_REASONING_PREVIEW = 1500


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("text"))
    return "" if content is None else str(content)


def _one_line(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _turn_name(user_text: str) -> str:
    if _capture_mode() == "metadata" or not user_text.strip():
        return "agent turn"
    return f"{_one_line(_redact(user_text), 70)}"


def _profile_name() -> str:
    home = _env("HERMES_HOME")
    return os.path.basename(home.rstrip("/\\")) if home else ""


def _short_tool(name: str) -> str:
    """``mcp__helpdesk__get_x`` → ``helpdesk.get_x``."""
    m = _MCP_NAME.match(name or "")
    return f"{m.group('server')}.{m.group('tool')}" if m else (name or "tool")


def _parse_args(args: Any) -> Any:
    if isinstance(args, str):
        try:
            return json.loads(args)
        except ValueError:
            return args
    return args


def _tool_targets(name: str, args: Any) -> list:
    """``[(real_tool_name, its_args)]``: a ``tool_call`` dispatcher is unwrapped to its calls."""
    args = _parse_args(args)
    if name in _META_TOOLS and isinstance(args, dict) and isinstance(args.get("calls"), list):
        calls = [c for c in args["calls"] if isinstance(c, dict) and c.get("name")]
        if calls:
            return [(str(c["name"]), c.get("arguments")) for c in calls]
    return [(name, args)]


def _tool_label(name: str, args: Any) -> str:
    return ", ".join(_short_tool(n) for n, _ in _tool_targets(name, args))


def _deep_parse(value: Any, depth: int = 0) -> Any:
    """JSON-encoded strings nested anywhere in a result become objects (no ``\\"`` walls)."""
    if depth > 12:
        return value
    if isinstance(value, str) and value[:1] in "{[":
        parsed = _parse_args(value)
        return _deep_parse(parsed, depth + 1) if isinstance(parsed, (dict, list)) else value
    if isinstance(value, dict):
        return {k: _deep_parse(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_parse(v, depth + 1) for v in value]
    return value


def _unwrap_result(result: Any) -> Any:
    """Parse JSON strings, and ``{"result": "<json>"}`` envelopes, so results read as objects."""
    result = _deep_parse(_parse_args(result))
    if isinstance(result, dict) and set(result) == {"result"} and isinstance(result["result"], (dict, list)):
        return result["result"]
    return result


def _tool_error(result: Any) -> str:
    """The failure message of a tool result, or "" when it succeeded."""
    data = _unwrap_result(result)
    if isinstance(data, dict):
        if data.get("error"):
            err = data["error"]
            return err if isinstance(err, str) else json.dumps(err, default=str, ensure_ascii=False)
        status = str(data.get("status") or "").lower()
        if status in {"error", "failed", "failure"}:
            return str(data.get("message") or status)
    return ""


def _call_parts(tc: Any) -> tuple:
    """``(name, arguments)`` from an OpenAI-style or Hermes tool-call object/dict."""
    if isinstance(tc, dict):
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        return str(fn.get("name") or ""), _parse_args(fn.get("arguments"))
    fn = getattr(tc, "function", None) or tc
    return str(getattr(fn, "name", "") or ""), _parse_args(getattr(fn, "arguments", None))


def _set_messages(span: Any, prefix: str, messages: list) -> None:
    """OpenInference ``llm.{input,output}_messages`` so Phoenix renders a chat view."""
    for i, msg in enumerate(messages):
        base = f"{prefix}.{i}.message"
        _set_attr(span, f"{base}.role", str(msg.get("role") or ""))
        text = _content_text(msg.get("content"))
        if text:
            _set_attr(span, f"{base}.content", _capture_text(text))
        for j, tc in enumerate(msg.get("tool_calls") or []):
            name, arguments = _call_parts(tc)
            _set_attr(span, f"{base}.tool_calls.{j}.tool_call.function.name", name)
            _set_attr(span, f"{base}.tool_calls.{j}.tool_call.function.arguments", _capture_json(arguments))


def _input_digest(messages: list) -> str:
    """One readable block per new message: who said what since the previous step."""
    rows = []
    for msg in messages:
        role = str(msg.get("role") or "")
        text = _content_text(msg.get("content")).strip()
        if role == "system":
            rows.append(f"[system prompt · {len(text)} chars]")
        elif role == "tool":
            label = _short_tool(str(msg.get("name") or msg.get("tool_name") or "tool"))
            rows.append(f"tool {label} →\n{text}")
        elif role == "assistant" and msg.get("tool_calls") and not text:
            continue
        elif text:
            rows.append(f"{'user' if role == 'user' else role}:\n{text}")
    return "\n\n".join(rows)


def _decision(output: Dict[str, Any]) -> tuple:
    """``(short_label, readable_text)`` of what one LLM step decided."""
    calls = [_call_parts(tc) for tc in output.get("tool_calls") or []]
    reasoning = _content_text(output.get("reasoning")).strip()
    content = _content_text(output.get("content")).strip()
    parts = []
    if reasoning:
        parts.append("Reasoning: " + (reasoning if len(reasoning) <= _REASONING_PREVIEW
                              else reasoning[:_REASONING_PREVIEW] + " …"))
    if calls:
        for name, arguments in calls:
            for target, target_args in _tool_targets(name, arguments):
                parts.append(f"→ {_short_tool(target)} {json.dumps(target_args, default=str, ensure_ascii=False)}")
        label = "→ " + ", ".join(_tool_label(n, a) for n, a in calls)
    else:
        label = "Trả lời"
    if content:
        parts.append(content)
    return label, "\n\n".join(parts)


# ── hooks ─────────────────────────────────────────────────────────────────────
def on_pre_llm_call(*, task_id: str = "", session_id: str = "", platform: str = "", model: str = "",
                    provider: str = "", api_mode: str = "", messages: Any = None,
                    turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    # Only legacy request-shaped calls carry an API ``messages`` list; the
    # turn-scoped pre_llm_call would otherwise open an orphan root.
    if not isinstance(messages, list):
        return
    project = _project_for(platform, session_id)
    tracer = _tracer_for_project(project)
    if tracer is None:
        return
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    with _STATE_LOCK:
        _get_or_start_state_locked(task_key, project=project, tracer=tracer, task_id=task_id, session_id=session_id,
                                   platform=platform, provider=provider, model=model, api_mode=api_mode,
                                   messages=messages, turn_id=turn_id, api_request_id=api_request_id)


def on_pre_llm_request(*, task_id: str = "", session_id: str = "", platform: str = "", model: str = "",
                       provider: str = "", base_url: str = "", api_mode: str = "", api_call_count: int = 0,
                       request_messages: Any = None, messages: Any = None, message_count: int = 0,
                       approx_input_tokens: int = 0, conversation_history: Any = None,
                       user_message: Any = None, turn_id: str = "", api_request_id: str = "",
                       request: Any = None, system_prompt: Any = None, **_: Any) -> None:
    project = _project_for(platform, session_id)
    tracer = _tracer_for_project(project)
    if tracer is None:
        return

    # The request body carries the model actually dispatched (mid-session switch,
    # fallback, middleware rewrite) — prefer it over the agent attribute.
    body_model = request["body"].get("model") if isinstance(request, dict) and isinstance(request.get("body"), dict) else None
    if isinstance(body_model, str) and body_model:
        model = body_model

    input_messages = request_messages or messages or conversation_history or (
        [{"role": "user", "content": user_message}] if user_message else [])
    req_key = _request_key(api_call_count)
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)

    with _STATE_LOCK:
        state = _get_or_start_state_locked(
            task_key, project=project, tracer=tracer, task_id=task_id, session_id=session_id, platform=platform,
            provider=provider, model=model, api_mode=api_mode, messages=input_messages, turn_id=turn_id,
            api_request_id=api_request_id)
        previous = state.generations.pop(req_key, None)
        if previous is not None:
            _end_span(previous)
        state.step_no += 1
        span = _start_child(state, name=f"{state.step_no}. Suy nghĩ…", kind=_KIND_LLM)
        if span is not None:
            _set_attr(span, _OI_LLM_MODEL, model)
            _set_attr(span, _OI_LLM_PROVIDER, provider)
            if isinstance(input_messages, list):
                # Only what is new since the previous step; step 1 also carries the system prompt.
                new = [m for m in input_messages[state.seen_messages:] if isinstance(m, dict)]
                if state.step_no == 1 and system_prompt:
                    new = [{"role": "system", "content": str(system_prompt)}] + new
                state.seen_messages = len(input_messages)
                _set_messages(span, "llm.input_messages", new)
                _set_attr(span, _OI_INPUT_VALUE, _capture_text(_input_digest(new)))
                _set_attr(span, _OI_INPUT_MIME, _MIME_TEXT)
            _set_metadata(span, {"step": state.step_no, "api_call": api_call_count, "provider": provider,
                                 "platform": platform, "api_mode": api_mode, "base_url": base_url,
                                 "message_count": message_count, "approx_input_tokens": approx_input_tokens})
            state.generations[req_key] = span
            state.step_of[id(span)] = state.step_no


def on_post_llm_call(*, task_id: str = "", session_id: str = "", provider: str = "", base_url: str = "",
                     api_mode: str = "", model: str = "", api_call_count: int = 0, assistant_message: Any = None,
                     response: Any = None, api_duration: float = 0.0, finish_reason: str = "", usage: Any = None,
                     assistant_content_chars: int = 0, assistant_tool_call_count: int = 0,
                     assistant_response: Any = None, turn_id: str = "", api_request_id: str = "",
                     response_model: Any = None, **_: Any) -> None:
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    # The turn-level post_llm_call (turn_finalizer: no api_call_count / api_request_id) fires once
    # after the whole loop, gate retries included, so it is what ends the turn's trace.
    if not api_call_count and not api_request_id and assistant_message is None:
        _finish_trace(task_key, output=_content_text(assistant_response) or None)
        return
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        generation = state.generations.pop(_request_key(api_call_count), None) if state else None
    if state is None or generation is None:
        return

    if isinstance(response_model, str) and response_model:
        model = response_model

    output = _serialize_assistant(assistant_message, assistant_response, assistant_content_chars,
                                  assistant_tool_call_count)
    label, text = _decision(output)
    tokens = _extract_tokens(response, usage)
    with _failsafe("end generation"):
        generation.update_name(f"{state.step_of.pop(id(generation), state.step_no)}. {label}")
        _set_attr(generation, _OI_LLM_MODEL, model)
        _set_attr(generation, _OI_OUTPUT_VALUE, _capture_text(text))
        _set_attr(generation, _OI_OUTPUT_MIME, _MIME_TEXT)
        _set_messages(generation, "llm.output_messages", [{
            "role": "assistant", "content": output.get("content"), "tool_calls": output.get("tool_calls"),
        }])
        for key, value in tokens.items():
            _set_attr(generation, key, value)
        _set_metadata(generation, {
            "tool_call_count": len(output.get("tool_calls", [])) or assistant_tool_call_count,
            "finish_reason": finish_reason or None,
            "api_duration_s": round(api_duration, 3) if api_duration and api_duration > 0 else None,
            "reasoning_chars": len(_content_text(output.get("reasoning"))) or None,
        })
        _end_span(generation)
    sdk = _load_sdk()
    with _STATE_LOCK:
        state.total_tokens += tokens.get(_OI_LLM_TOKENS_TOTAL, 0)
        # Tools this step asked for nest under it.
        state.step_context = sdk["set_span_in_context"](generation) if sdk and output.get("tool_calls") else None


def on_pre_tool_call(*, tool_name: str = "", args: Any = None, task_id: str = "",
                     session_id: str = "", tool_call_id: str = "",
                     turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        targets = _tool_targets(tool_name, args)
        span = _start_child(state, name=_tool_label(tool_name, args), kind=_KIND_TOOL,
                            context=state.step_context)
        if span is None:
            return
        state.tool_count += 1
        # A tool_call dispatcher shows its real target and that target's arguments.
        real_name, real_args = targets[0] if len(targets) == 1 else (tool_name, _parse_args(args))
        _set_attr(span, _OI_TOOL_NAME, real_name)
        _set_attr(span, _OI_INPUT_VALUE, _capture_json(real_args))
        _set_attr(span, _OI_INPUT_MIME, _MIME_JSON)
        _set_metadata(span, {"tool_name": real_name, "dispatcher": tool_name if real_name != tool_name else None,
                             "tool_call_id": tool_call_id})
        if tool_call_id:
            state.tools[tool_call_id] = span
        else:
            state.pending_tools_by_name.setdefault(tool_name, []).append(span)


def on_post_tool_call(*, tool_name: str = "", args: Any = None, result: Any = None,
                      task_id: str = "", session_id: str = "", tool_call_id: str = "",
                      turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        span = state.tools.pop(tool_call_id, None) if tool_call_id else None
        queue = state.pending_tools_by_name.get(tool_name) if span is None else None
        if queue:
            span = queue.pop(0)
            if not queue:
                state.pending_tools_by_name.pop(tool_name, None)
    if span is None:
        return
    error = _tool_error(result)
    with _failsafe("end tool"):
        if error:
            # A failure reads as its message, as plain text.
            _set_attr(span, _OI_OUTPUT_VALUE, _capture_text(f"ERROR: {error}"))
            _set_attr(span, _OI_OUTPUT_MIME, _MIME_TEXT)
        else:
            _set_attr(span, _OI_OUTPUT_VALUE, _capture_json(_unwrap_result(result)))
            _set_attr(span, _OI_OUTPUT_MIME, _MIME_JSON)
        if error:
            # Red in Phoenix, with the reason on the span.
            _mark_error(span, _one_line(error, 200))
            with _STATE_LOCK:
                state.tool_errors += 1
        _end_span(span)


def on_api_request_error(*, task_id: str = "", session_id: str = "", api_call_count: int = 0,
                         api_duration: float = 0.0, status_code: Any = None, retry_count: Any = None,
                         max_retries: Any = None, retryable: Any = None, reason: Any = None, error: Any = None,
                         turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    """Close (as ERROR) the open generation for a failed API request; a
    non-retryable failure also finishes the turn since the loop is about to unwind."""
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        generation = state.generations.pop(_request_key(api_call_count), None) if state else None
    if state is None:
        return

    error = error if isinstance(error, dict) else {}
    error_type, error_message = str(error.get("type") or ""), str(error.get("message") or "")
    error_metadata = {
        "error": True, "error_type": error_type, "error_message": _capture_text(error_message),
        "status_code": status_code, "retry_count": retry_count, "max_retries": max_retries,
        "retryable": retryable, "reason": str(reason) if reason else None,
        "api_duration_s": round(api_duration, 3) if api_duration and api_duration > 0 else None,
    }
    if generation is not None:
        with _failsafe("error generation"):
            _mark_error(generation, error_type or "api_request_error")
            _set_metadata(generation, error_metadata)
            _end_span(generation)

    if retryable is False:
        _finish_trace(task_key, output={"error": error_metadata}, error=True,
                      status_message=error_type or "api_request_error")
    else:
        state.last_updated_at = time.time()


def on_session_finalize(*, session_id: str = "", reason: str = "", **_: Any) -> None:
    """Session-end boundary: close still-open traces and flush. A turn ending on a
    tool-only or empty final response never reaches ``_finish_trace``; its root
    would dangle until eviction and queued spans could be lost on exit."""
    with _PROVIDER_LOCK:
        providers = list(_PROVIDERS.values())
    if not providers:
        return

    fragments = (f"session:{session_id}", f"task:{session_id}")
    with _STATE_LOCK:
        keys = [k for k in _TRACE_STATE if not session_id or k == session_id or any(f in k for f in fragments)]
        # Drop any stashed gate verdict that never got a root (turn failed pre-LLM).
        if session_id:
            _PENDING_GATES.pop(session_id, None)
        else:
            _PENDING_GATES.clear()
    for key in keys:
        _finish_trace(key)
    for provider in providers:
        with _failsafe("finalize flush"):
            provider.force_flush()

    # Shut down only at true process exit (not /new, /reset, session expiry: the
    # providers must keep exporting for later sessions).
    if reason == "shutdown":
        _shutdown_all()


def on_subagent_start(*, parent_turn_id: str = "", parent_subagent_id: Any = None,
                      child_session_id: Any = None, child_subagent_id: Any = None,
                      child_role: str = "", child_goal: Any = None, **_: Any) -> None:
    if not child_session_id:
        return
    with _STATE_LOCK:
        state = _state_for_turn(parent_turn_id)
        if state is None:
            return
        span = _start_child(state, name=f"Subagent: {child_role or 'delegate'}", kind=_KIND_AGENT)
        if span is None:
            return
        _set_attr(span, _OI_INPUT_VALUE, _capture_json(child_goal))
        _set_metadata(span, {"child_session_id": child_session_id, "child_subagent_id": child_subagent_id,
                             "child_role": child_role, "parent_subagent_id": parent_subagent_id})
        state.subagents[str(child_session_id)] = span


def on_subagent_stop(*, parent_turn_id: str = "", child_session_id: Any = None, child_role: str = "",
                     child_summary: Any = None, child_status: Any = None,
                     tool_call_history: Any = None, duration_ms: Any = None, **_: Any) -> None:
    if not child_session_id:
        return
    with _STATE_LOCK:
        state = _state_for_turn(parent_turn_id)
        span = state.subagents.pop(str(child_session_id), None) if state else None
    if span is None:
        return
    with _failsafe("end subagent"):
        _set_attr(span, _OI_OUTPUT_VALUE, _capture_json(child_summary))
        _set_metadata(span, {"child_role": child_role, "status": child_status, "duration_ms": duration_ms,
                             "tool_call_count": len(tool_call_history) if isinstance(tool_call_history, list) else None})
        _end_span(span)


def on_pre_turn_gate(*, gate: str = "", session_id: str = "", platform: str = "", input_text: Any = "",
                     disable_tools: Any = None, p_needs_tools: Any = None, skip_confidence: Any = None,
                     model: str = "", status: str = "", duration_ms: Any = None, **_: Any) -> None:
    """A detachable pre-turn tool gate (JEV) evaluated whether the turn needs tools,
    BEFORE the LLM call. The turn's root span does not exist yet, so stash the verdict
    by session_id; ``_start_root`` attaches it as a GUARDRAIL child of the turn."""
    if not session_id:
        return
    # Only stash when tracing is actually active for this flow's project.
    if _tracer_for_project(_project_for(platform, session_id)) is None:
        return
    payload = {"gate": gate, "input_text": input_text, "disable_tools": disable_tools,
               "p_needs_tools": p_needs_tools, "skip_confidence": skip_confidence,
               "model": model, "status": status, "duration_ms": duration_ms, "_ts": time.time()}
    with _STATE_LOCK:
        _PENDING_GATES[session_id] = payload
        over = len(_PENDING_GATES) - _MAX_PENDING_GATES
        for key, _v in sorted(_PENDING_GATES.items(), key=lambda kv: kv[1].get("_ts", 0))[:max(over, 0)]:
            _PENDING_GATES.pop(key, None)


def on_claim_gate(*, task_id: str = "", session_id: str = "", turn_id: str = "", answer: Any = "",
                  verdict: str = "", detail: Any = None, rule_hit: Any = None, judge: str = "",
                  judge_error: str = "", duration_ms: Any = None, nudge: Any = None, attempt: Any = None,
                  judge_input: Any = None, judge_reply: Any = None, judge_started: Any = None,
                  judge_ended: Any = None, **_: Any) -> None:
    """The chat claim gate (agent.claim_stop) checked a final answer before the turn ends: one
    GUARDRAIL span with the verdict, how the judge went, and the nudge sent back if any."""
    with _STATE_LOCK:
        state = _TRACE_STATE.get(_trace_key(task_id, session_id, turn_id=turn_id)) or _state_for_turn(turn_id)
    if state is None:
        return
    ms = duration_ms if isinstance(duration_ms, (int, float)) and duration_ms > 0 else 0
    outcome = f"{verdict} → nudge" if nudge else "OK"
    judge_note = f" · judge {judge}" if judge and judge != "ok" else ""
    span = _start_child(state, name=f"Claim gate: {outcome}{judge_note}", kind=_KIND_GUARDRAIL,
                        start_time=time.time_ns() - int(ms * 1_000_000))
    if span is None:
        return
    with _failsafe("claim gate span"):
        _set_attr(span, _OI_INPUT_VALUE, _capture_text(_content_text(answer)))
        _set_attr(span, _OI_INPUT_MIME, _MIME_TEXT)
        _set_attr(span, _OI_OUTPUT_VALUE, _capture_text(str(nudge)) if nudge else "OK — answer accepted")
        _set_attr(span, _OI_OUTPUT_MIME, _MIME_TEXT)
        _set_metadata(span, {"verdict": verdict or "OK", "detail": detail, "rule_hit": rule_hit,
                             "judge": judge, "judge_error": judge_error or None,
                             "duration_ms": round(ms, 1) if ms else None, "attempt": attempt})
        if judge_input is not None:
            _emit_judge_span(state, span, judge_input=judge_input, judge_reply=judge_reply,
                             judge=judge, judge_error=judge_error, started=judge_started, ended=judge_ended)
        _end_span(span)


def _emit_judge_span(state: TraceState, gate_span: Any, *, judge_input: Any, judge_reply: Any, judge: str,
                     judge_error: str, started: Any, ended: Any) -> None:
    """The judge's own LLM call, nested under the claim gate: what it read and what it answered."""
    sdk = _load_sdk()
    if sdk is None:
        return
    start_ns = int(started * 1e9) if isinstance(started, (int, float)) else None
    end_ns = int(ended * 1e9) if isinstance(ended, (int, float)) else None
    name = "Judge" + (f" ({judge})" if judge and judge != "ok" else "")
    span = _start_child(state, name=name, kind=_KIND_LLM, context=sdk["set_span_in_context"](gate_span),
                        start_time=start_ns)
    if span is None:
        return
    try:
        from agent.claim_stop import _JUDGE_INSTRUCTIONS as instructions
    except Exception:
        instructions = ""
    reply = _parse_args(judge_reply) if judge_reply else None
    _set_messages(span, "llm.input_messages", [{"role": "system", "content": instructions},
                                               {"role": "user", "content": str(judge_input)}])
    _set_attr(span, _OI_INPUT_VALUE, _capture_text(str(judge_input)))
    _set_attr(span, _OI_INPUT_MIME, _MIME_TEXT)
    if isinstance(reply, (dict, list)):
        _set_attr(span, _OI_OUTPUT_VALUE, _capture_json(reply))
        _set_attr(span, _OI_OUTPUT_MIME, _MIME_JSON)
    else:
        _set_attr(span, _OI_OUTPUT_VALUE, _capture_text(str(judge_reply or f"ERROR: {judge_error or judge}")))
        _set_attr(span, _OI_OUTPUT_MIME, _MIME_TEXT)
    if judge_reply:
        _set_messages(span, "llm.output_messages", [{"role": "assistant", "content": str(judge_reply)}])
    if judge != "ok":
        _mark_error(span, judge_error or judge)
    with _failsafe("judge span end"):
        span.end(end_time=end_ns)


def register(ctx) -> None:
    # Both hook-name variants so the plugin works across Hermes versions:
    # *_api_request fire per API call (preferred); *_llm_call once per turn.
    hooks = (
        ("pre_api_request", on_pre_llm_request), ("post_api_request", on_post_llm_call),
        ("api_request_error", on_api_request_error), ("pre_llm_call", on_pre_llm_call),
        ("post_llm_call", on_post_llm_call), ("pre_tool_call", on_pre_tool_call),
        ("post_tool_call", on_post_tool_call), ("on_session_finalize", on_session_finalize),
        ("on_session_end", on_session_finalize), ("subagent_start", on_subagent_start),
        ("subagent_stop", on_subagent_stop), ("pre_turn_gate", on_pre_turn_gate),
        ("claim_gate", on_claim_gate),
    )
    for name, fn in hooks:
        ctx.register_hook(name, fn)
