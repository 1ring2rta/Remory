# Codex

This example connects Remory to the compaction and generation handlers of a
custom Responses provider. Codex runs the tools; the provider serves the Qwen
actor and carries residual memory between requests.

## Run the example

Start Remory with `./deploy.sh`, then run these commands from the repository root
with Node 22+:

```bash
.remory/venv/bin/python examples/quickstart.py --write-input /tmp/remory-input.json
node integrations/codex/example.mjs /tmp/remory-input.json
```

The Python example renders a conversation with the Qwen tokenizer. The JavaScript
example compacts it, saves a Responses compaction item, and passes that item into
the next generation. Use `REMORY_URL`, `REMORY_API_KEY`, and `REMORY_SESSION` to
configure the JavaScript client.

## Provider integration

[`bridge.mjs`](bridge.mjs) exports `compactResponse`, `readCompactionItem`, and
`generateFromItems`. Use a [`RemoryClient`](../client.mjs) bound to a stable session ID:

```js
import { RemoryClient } from "../client.mjs";
import { compactResponse, generateFromItems } from "./bridge.mjs";

const client = new RemoryClient({ url: "http://127.0.0.1:8421",
  apiKey: process.env.REMORY_API_KEY, sessionId });

// In POST /responses/compact, after generating the new summary:
const compacted = await compactResponse(client, {
  prefixIds, historyIds, summaryIds, previous,
});

// In POST /responses:
const generated = await generateFromItems(client, request.input, {
  render: renderNativeQwenItems,
  maxNewTokens: 1024,
});
```

The provider implements `renderNativeQwenItems(items, { continuation })` using the
actor's chat template. Before compaction, render the full prompt with system
instructions and tools. After compaction, render only the retained/new messages
and generation prompt; Remory restores the prefix and summary. Preserve native
tool calls, their IDs, and results.

`compactResponse` returns a `compaction` item whose `encrypted_content` field
contains an opaque `remory.v1.rm_…` database handle. `generateFromItems` finds the
latest such item and restores the memory. The handle is a local reference and
can only be resolved by this provider.

For the next compaction, use `readCompactionItem` to obtain `previous` and send
only the newly removed history. Save a successful compaction response with the
transcript before discarding the original history. Keep the session ID stable
across compactions and resume; branches can retain separate immutable handles.

A complete provider also implements summary generation, native output/tool
parsing, and Responses streaming. Remory serves `/v1/compact` and `/v1/generate`,
so the Codex base URL must point to that provider. The example has been tested
against the live Remory backend; full interactive Codex sessions remain untested.
Check the Codex version's remote-compaction dispatch when adding the provider.

References: [Codex custom providers](https://developers.openai.com/codex/config-advanced/),
[Responses compaction](https://developers.openai.com/api/docs/guides/compaction),
[compaction source](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs).
