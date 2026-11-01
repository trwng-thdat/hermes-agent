---
sidebar_position: 17
title: "Execution Ledger & Claim Gate"
description: "How chat answers are kept from reporting actions as done without execution evidence: per-call receipts, three-state outcomes, duplicate-write prevention and a deterministic final-answer gate."
---

# Execution Ledger & Claim Gate

A chat agent must never tell the user an action is done (a tag updated, an email sent) unless a
tool call actually did it. Hermes enforces this in code, with no extra LLM call:

1. every **write** tool result gets an **execution receipt** (`agent/execution_ledger.py`);
2. receipts for the current turn form the **execution ledger**, kept by the runtime, never by the model;
3. before a text answer ends the turn, the **claim gate** (`agent/claim_stop.py`) checks any
   "done" claim in it against the ledger.

Scope: `agent.claim_gate: auto` (the default) turns this on for the ai-claw dashboard chat, TUI and
desktop; cron, subagents, kanban workers and messaging platforms stay off. `HERMES_CLAIM_GATE` wins
over config. Receipts are written for every write call regardless of the gate.

## Flow

```
model → tool call ──► before_execute ──► execute ──► after_execute ──► result + _receipt ──► model
                       │ duplicate?                    │ classify in code
                       └─ same write SUCCEEDED → old receipt, not executed
                          same write UNKNOWN, no read since → refused

model → text answer ──► claim gate ──► OK ─────────────► user
                         │ unbacked claim, retries left → retract + nudge (same turn, ledger kept)
                         └ unbacked claim, no retries   → replace with receipts summary
```

## Write tools

A tool is a write when its name carries a write verb (`create`, `update`, `delete`, `remove`,
`send`, `schedule`, `cancel`, `add`). For a `helpdesk` MCP server the HTTP verb in the generated
name decides (`post_…`, `put_…`, `patch_…`, `delete_…`, …). Reads never get receipts.

## Three states

Each write result is classified by `classify()`:

| Result | Status |
|---|---|
| 2xx body with nothing wrong in it (authoritative response) | `SUCCEEDED`, accepted **without a read-back** |
| 2xx body with `success: false`, `ok: false`, `statusCode ≥ 400`, `status: FAILED`, non-empty `error` / `errors` | `FAILED` (HTTP 200 is not success by itself) |
| Response echoes a requested field with another value (e.g. asked `tagID: vip`, got `normal`) | `FAILED` — postcondition not met |
| `4xx: <message>` refusal | `FAILED` |
| 202 / `status: pending|queued|processing` | `UNKNOWN` — accepted, not completed |
| timeout, 5xx / "internal error", transport lost mid-call, `outcome_uncertain` | `UNKNOWN` — it may have landed |
| anything else from the tool | `FAILED` |

The postcondition check compares only short scalars and list items the response echoes under
the same key (top level or under `data`); long text and keys the response does not echo are not
compared. It never issues a request of its own.

A write-capable MCP `tools/call` that times out after dispatch is reported as
`outcome_uncertain` (UNKNOWN) and never replayed (`tools/mcp_tool_handlers._handle_write_timeout`).

The receipt is merged into the result the model reads, so the agent knows the real outcome before
it answers:

```json
{"result": "{\"status\": \"OK\", \"data\": {...}}",
 "_receipt": {"operation_id": "call_1", "tool": "mcp__helpdesk__post_customer_by_customer_id_tags",
              "status": "SUCCEEDED", "verified": true, "evidence": "authoritative API response"}}
```

`_detect_tool_failure` honours `_receipt`, so the TUI, Phoenix traces and the backend's tool
analytics see the same verdict.

## Duplicate writes and recovery

The ledger is per turn and survives claim-gate retries, so earlier successes are kept and only the
unresolved operation is worked on. An operation is identified by tool + canonical arguments.

- Same write already `SUCCEEDED` this turn → not executed; the earlier receipt comes back.
- Same write `UNKNOWN` → refused until any read tool has run (the agent checked the state).
- A retried add/create refused with "already exists" after an `UNKNOWN` attempt → `SUCCEEDED`
  (the first attempt landed).
- `FAILED` → may be retried; a later success of the same operation resolves it.

This prevents duplicates within a turn. Across turns it relies on the API: Helpdesk routes do not
accept an idempotency key today.

## Claim gate

`check_answer()` finds completion claims ("đã gửi", "đã gắn tag", "thành công", "xong", "sent",
"successfully", "done"…), skipping questions, negated clauses ("chưa thể cập nhật … thành công",
"đã gửi không thành công") and conditional or future ones ("khi xong", "sẽ gửi", "once done").

| Situation | Decision |
|---|---|
| No claim | pass |
| The user asked a question ("đã gửi chưa?") and no write ran | pass (answered from history) |
| Claim with no `SUCCEEDED` write of a matching kind (send / create / update / delete; generic claims match any write) | `UNBACKED_CLAIM` |
| Claim while a write is `FAILED` / `UNKNOWN` and the answer never mentions a failure or uncertainty | `UNREPORTED_FAILURE` |
| A subagent did the work | pass (its writes are not in this transcript) |

On a hit with retries left (`agent.max_claim_nudges`, default 2) the answer is retracted and a nudge
listing the ledger (SUCCEEDED — do not repeat; FAILED — with errors; UNKNOWN — check state first)
continues the **same** turn. When retries or the iteration budget are spent, the answer is
replaced by a summary that states only what the receipts prove.

Once a write ran in a turn, the answer is withheld from the token stream and delivered whole in
`message.complete`, so a retracted answer is never shown to the user. Turns without a write stream
as before.

## Latency

The gate and the receipts are pure code over the current turn's transcript: no LLM call and no
network request (sub-millisecond per write, a few milliseconds per answer). The only added cost is
a retry when the agent really did claim something unsupported. The previous LLM grounding judge
(up to 20 s per suspect answer, fail-open on timeout) is gone.

## Observability

The `claim_gate` plugin hook fires once per checked answer with the verdict, the action
(`nudge` / `replace`), the receipts it read and the timing; the Phoenix plugin renders it as a
`Claim gate: <verdict> → <action>` GUARDRAIL span.

## Limits

- "Required" actions are the writes the agent attempted plus what its answer claims. If the user
  asked for three things and the agent did two and says only "xong", the gate cannot know about the
  third.
- Claim detection is pattern-based (Vietnamese and English).
- No asynchronous completion check for 202 / pending operations: they stay `UNKNOWN` and the
  agent must say so.
- No automatic read-back for `UNKNOWN`: the agent must call a read tool before a retry is allowed.
- Duplicate prevention is per turn; cross-turn idempotency needs API support.
