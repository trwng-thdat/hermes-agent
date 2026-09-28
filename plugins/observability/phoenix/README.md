# Phoenix Observability Plugin

Traces Hermes agent turns, LLM calls, and tool usage to
[Arize Phoenix](https://github.com/Arize-ai/phoenix) as OpenTelemetry spans
annotated with [OpenInference](https://github.com/Arize-ai/openinference)
semantic conventions, exported over OTLP/HTTP.

This plugin ships bundled with Hermes but is **opt-in** — it only loads when you
explicitly enable it. Without the OTel SDK or `HERMES_PHOENIX_ENDPOINT` the hooks
no-op silently (fail-open).

## Three projects, by turn origin

Each agent turn is routed to one of three Phoenix **projects**
(`openinference.project.name`), decided from the `platform` the turn runs under:

| Flow | How it runs | Phoenix project |
|------|-------------|-----------------|
| **agent-chat** | interactive chat (dashboard WS / CLI / gateway) | `agent-chat` |
| **wiki** | agentic-ingest agent runs via `/v1/runs` (`api_server` platform) | `wiki` |
| **cron** | cron-scheduled turns (`cron` platform) | `cron` |

The mapping is: `platform == "cron"` → cron, `platform == "api_server"` → wiki,
everything else → agent-chat. Override any project name with the env vars below.

## Enable

```bash
pip install 'hermes-agent[otlp]'          # OpenTelemetry SDK + OTLP/HTTP exporter
hermes plugins enable observability/phoenix
```

## Run a local Phoenix

```bash
# Phoenix listens on 6006 (UI + OTLP/HTTP collector at /v1/traces)
docker run -p 6006:6006 arizephoenix/phoenix:latest
# open http://localhost:6006
```

## Required configuration

Set in `~/.hermes/.env` (or the process environment):

```bash
HERMES_PHOENIX_ENDPOINT=http://localhost:6006/v1/traces
```

For **Phoenix Cloud** instead of a local instance:

```bash
HERMES_PHOENIX_ENDPOINT=https://app.phoenix.arize.com/v1/traces
HERMES_PHOENIX_HEADERS=api_key=<your-phoenix-api-key>
```

## Verify

```bash
hermes plugins list                  # observability/phoenix should show "enabled"
hermes chat -q "hello"               # then open http://localhost:6006 → project "agent-chat"
```

## Optional tuning

```bash
HERMES_PHOENIX_PROJECT_CHAT=agent-chat   # rename the interactive-chat project
HERMES_PHOENIX_PROJECT_WIKI=wiki         # rename the wiki-ingest project
HERMES_PHOENIX_PROJECT_CRON=cron         # rename the cron project
HERMES_PHOENIX_SERVICE_NAME=hermes-agent # service.name resource attribute
HERMES_PHOENIX_HEADERS=api_key=...       # comma-separated k=v OTLP export headers
HERMES_PHOENIX_CAPTURE=sanitized         # content capture mode (see below)
HERMES_PHOENIX_MAX_CHARS=12000           # max chars per captured field (default 12000)
HERMES_PHOENIX_DEBUG=true                # verbose plugin logging
```

## Capture modes

`HERMES_PHOENIX_CAPTURE` controls how much *content* (prompts, responses, tool
arguments/results) is exported. Structural metadata — IDs, roles, tool names,
token counts, model, timing — is always captured.

| mode | behavior |
|------|----------|
| `metadata` | No content. Each content field becomes a shape/size stub. |
| `sanitized` | **(default)** Content exported after secret-pattern redaction and truncation. Redaction runs *before* truncation. |
| `full` | Raw content, truncated only. Traces will contain whatever passed through the conversation. |

`sanitized` is pattern-based defense in depth, not a DLP guarantee. For shared
Phoenix projects, prefer `metadata`.

## Span shape

- **agent turn** — root span, `openinference.span.kind = CHAIN`, `session.id` set
  for Phoenix session grouping; input = last user message, output = final answer.
- **LLM call N** — `kind = LLM`, with `llm.model_name`, `llm.provider`,
  `llm.token_count.prompt/completion/total`, input messages, output message.
- **Tool: `<name>`** — `kind = TOOL`, with `tool.name`, input args, output result.
- **Subagent: `<role>`** — `kind = AGENT`, delegated child runs under the turn.

Failed model requests close their LLM span with an ERROR status; non-retryable
failures also finish the turn span. Session end/finalize closes any still-open
turn spans and force-flushes, so interrupted or tool-only turns don't dangle.

## Disable

```bash
hermes plugins disable observability/phoenix
```
