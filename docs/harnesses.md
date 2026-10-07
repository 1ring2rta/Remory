# Codex integration example

Connect Remory at a custom Responses provider's **compaction** and **generation**
handlers. Codex keeps control of tools and the conversation. The provider renders
Codex input with the Qwen actor's native chat template and translates actor output
back into Responses messages, function calls, and streaming events.

## Try the two connections

With `./deploy.sh` running, execute from the repository root (Node 22+):

```bash
.remory/venv/bin/python examples/quickstart.py --write-input /tmp/remory-input.json
node integrations/codex/example.mjs /tmp/remory-input.json
```

The Python example writes native token IDs. The JavaScript example calls the
compact handler, carries the returned item into a later request, and generates
with the restored memory. It exercises the live Remory service; it does not start
an interactive Codex session. For a different service address, set `REMORY_URL`;
for a custom runtime directory, set `REMORY_API_KEY` from its key file.

## In your Responses provider

[`bridge.mjs`](../integrations/codex/bridge.mjs) supplies two functions:

```js
// Inside POST /responses/compact, after producing a new text summary:
const compacted = await compactResponse(client, {
  prefixIds, historyIds, summaryIds, previous,
});
// Return compacted; commit the transcript change only after this succeeds.

// Inside POST /responses, before parsing native actor output into Responses items:
const generated = await generateFromItems(client, request.input, {
  render: renderNativeQwenItems,
  maxNewTokens: 1024,
});
```

`renderNativeQwenItems(items, { continuation })` returns token IDs. Before the first
compaction it renders the full prompt, including system instructions and tools.
After compaction it renders only the retained/new tail and generation prompt:
Remory restores the saved prefix and summary. Keep the original prefix immutable.
Do not flatten tool calls and results into arbitrary strings.

The compact response carries an opaque `remory.v1.rm_…` database reference in a
Responses `compaction` item. On the next request, `generateFromItems` finds the
latest item and restores its handle. The field name `encrypted_content` belongs
to the Responses format; this reference is local state, not OpenAI ciphertext.
Foreign compaction items are rejected.

For repeated compaction, recover the prior handle with `readCompactionItem` and
pass it as `previous`; send only newly removed history. Bind both handlers to the
same stable Remory session ID, including after resume and when branching. Preserve
retained items after the checkpoint, and keep the previous transcript on errors.

## Provider boundary

The included module implements the residual-memory connections. A complete Codex
provider must also implement native rendering, summary generation, tool parsing,
and Responses streaming. Remory's `/v1/compact` and `/v1/generate` endpoints are
a different API, so pointing Codex's base URL directly at port 8421 is insufficient.

Check the Codex version's remote-compaction dispatch when wiring a custom provider;
a tool/MCP server does not intercept that model request. This integration uses a
locally served Qwen actor, whose embeddings can accept the soft memory. Interactive
Codex sessions have not been certified by the repository tests.

Official references: [custom providers](https://developers.openai.com/codex/config-advanced/),
[Responses compaction](https://developers.openai.com/api/docs/guides/compaction),
[Codex compaction source](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs).
