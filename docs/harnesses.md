# Connect a coding harness

The shared seam has three steps: capture the exact native source before the
harness discards it; commit a residual checkpoint conditioned on the new summary;
restore that checkpoint at the provider's next actor request. Tool execution stays
inside the harness. Use the same actor tokenizer and renderer for all three.

Included modules are integration helpers. They do not replace complete Responses,
Messages, or Chat Completions providers. The examples intentionally require a
native rendering callback rather than flattening tool transcripts into strings.

## Codex

Use a custom Responses provider with your residual-capable actor. In its compact
handler, render Codex's input items and tools with the actor's template, produce
the new summary, then call `compactResponse` from `integrations/codex/bridge.mjs`.
Its output contains an opaque Remory compaction item. On later requests, detect
that item with `readCompactionItem`, recover its handle, render only the retained
tail/new input, and call `client.generate`. Convert the native actor output back
into Responses messages/function calls and implement the streaming events Codex
expects. Carry all relevant non-compaction output items as part of your provider.

The `encrypted_content` name belongs to the Responses wire format. Remory puts a
database handle there, not OpenAI ciphertext. Send it only to this custom provider,
and bind each request to the same stable session ID. Codex versions differ in
when they select remote compaction; check or patch that dispatch when connecting
a custom provider. A tool/MCP integration does not intercept model compaction.

Primary references: [Codex custom providers](https://developers.openai.com/codex/config-advanced/),
[Responses compaction](https://developers.openai.com/api/docs/guides/compaction),
[Codex compaction source](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs).

## Claude Code

Use an Anthropic-compatible gateway that runs the supported residual actor.
Capture the exact tokenized request before compaction, keyed by a stable session
identity. `PreCompact` exposes the transcript path; native transcript text alone
is not a substitute for the gateway's tool definitions and native prompt template.
After `PostCompact` supplies `compact_summary`, call `afterCompact` in
`integrations/claude-code/bridge.mjs` and persist the descriptor in the gateway.
Only acknowledge the transition as residual-ready once this succeeds. The next
gateway generation uses `generateWithCheckpoint`. Keep the captured raw source
until that transition commits, so a backend failure can be retried or surfaced
without silently claiming residual memory survived.

`PostCompact` is observational and cannot change the compaction result. Its
output cannot inject memory into Claude. The gateway needs the Messages protocol,
streaming, native tool parsing, session mapping, and the residual actor route;
the included bridge is the compaction/inference part of that provider. A hook-only
installation using Anthropic's hosted Claude model does not support soft memory.

Primary references: [Claude Code hooks](https://code.claude.com/docs/en/hooks),
[LLM gateways](https://code.claude.com/docs/en/llm-gateway).

## OpenCode (`opencode-ai/opencode`)

This is the archived Go project that continued as Crush. It is distinct from the
TypeScript project now also called OpenCode. The included Go module targets the
former's code structure, without assuming the latter's plugin hooks.

The compaction seam is `agent.Summarize` in `internal/llm/agent/agent.go`. After
the new summary is produced and before creating the continuation session, call
`Client.Compact` from `integrations/opencode/remory.go`. Pass native prefix,
removed-history, and summary IDs. Persist its handle on the new session together
with the stable Remory session identity; native OpenCode creates a new session ID
at this point, so do not accidentally switch the memory owner. Preserve a separate
branch checkpoint when forking. Provider generation then calls `Client.Generate`.

The Go module uses only the standard library. Map generated native Qwen tool calls
back into OpenCode's provider response type, leaving permission checks and tool
execution unchanged. On an error, return before committing the replacement session.

Primary references: [repository](https://github.com/opencode-ai/opencode),
[Summarize implementation](https://github.com/opencode-ai/opencode/blob/main/internal/llm/agent/agent.go).

## Pi

`installRemoryCompaction` in `integrations/pi/extension.mjs` registers
`session_before_compact`. Supply:

1. `clientForSession(ctx)`: an HTTP client bound to a durable memory session ID.
2. `buildRequest(event, ctx)`: generate the native summary, render the exact
   removed span, and return `summary`, `prefixIds`, `historyIds`, `summaryIds`,
   plus the previous handle when present.

The extension returns Pi's compaction result with the original
`firstKeptEntryId`/`tokensBefore` and `details.remory` containing the checkpoint.
It explicitly cancels compaction on failure. Your residual model provider reads
the latest active branch's compaction details on **every** generation, including
resume and `/tree` navigation, and invokes `client.generate` with the retained
tail. Never store only one process-global handle: it leaks state across branches.
When moving to a different native session ID, preserve or explicitly remap the
Remory session identity along with its checkpoint.

Primary references: [Pi extension types](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/src/core/extensions/types.ts),
[compaction](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/compaction.md),
[custom providers](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/custom-provider.md).

## Verification status

The package tests exercise the sidecar, backend protocol, compaction state, and
tiny real Qwen inference. JavaScript tests exercise carrier and hook behavior;
Go tests exercise HTTP serialization. Full interactive sessions in the four
coding clients have not been run against this release. Treat provider wiring as
integration work and validate tool execution, interrupted compaction, resume,
and branch behavior before relying on it for long agent sessions.
