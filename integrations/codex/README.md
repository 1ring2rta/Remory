# Codex

This example connects Remory to the compaction and generation handlers of a
custom Responses provider. Codex runs the tools; all model calls go to the same
SGLang server. The provider replaces compact with summary generation followed
by residual encoding, and carries the resulting checkpoint between requests.

## Run the example

Start SGLang with `python -m remory.launch_server` after
[installing the runtime](../../docs/deployment.md#install), then run these commands
from the repository root with Node 22+:

```bash
source .remory/venv/bin/activate
python examples/quickstart.py --write-input /tmp/remory-input.json
node integrations/codex/example.mjs /tmp/remory-input.json
```

The Python example generates a summary with SGLang and saves the native token
IDs. The JavaScript example adds residual memory, saves a Responses compaction
item, and continues through SGLang `/generate`. Use `REMORY_URL` and
`REMORY_SESSION` to configure the JavaScript client.

For GLM, start the [GLM server](../../docs/deployment.md#glm-53-flash) and pass
`--model-path zai-org/GLM-5.3-Flash --remory-checkpoint mocoV3/GLM-5.3-Flash-REMORY-1.24B`
to the Python example. It renders GLM's native tokens; the same JavaScript bridge
then uses them unchanged. In your provider, use `GlmChatTemplate` to extract the
completed answer from GLM's reasoning before encoding the summary, and omit the
Qwen-specific JSON schema.

## Provider integration

[`bridge.mjs`](bridge.mjs) exports `compactResponse`, `readCompactionItem`, and
`generateFromItems`. Use a [`RemoryClient`](../client.mjs) bound to a stable session ID:

```js
import { RemoryClient } from "../client.mjs";
import { compactResponse, generateFromItems } from "./bridge.mjs";

const client = new RemoryClient({ url: "http://127.0.0.1:8421", sessionId });

// Replace the provider's compact handler.
async function compact(request) {
  const summary = await client.generate({
    ...renderSummaryRequest(request), // inputIds, or handle + continuationIds
    maxNewTokens: 2048,
    sampling: { json_schema: JSON.stringify(summarySchema) },
  });
  if (summary.meta_info.finish_reason.type !== "stop") {
    throw new Error("Summary did not complete; retain the current context");
  }
  return compactResponse(client, {
    ...renderCompactionSource(request), // prefixIds, historyIds, previous
    summaryIds: encodeSummary(summary.text),
  });
}

// In POST /responses:
const generated = await generateFromItems(client, request.input, {
  render: renderNativeQwenItems,
  maxNewTokens: 1024,
});
```

The provider implements the rendering helpers with the actor's chat template.
`renderSummaryRequest` appends the summary instruction to the active context;
after a previous compaction, it uses that handle and the new messages.
`renderCompactionSource` selects the removed history without the summary
instruction. `encodeSummary` tokenizes the completed summary without chat markers.
The released checkpoint includes the summary prompt and JSON schema.

`renderNativeQwenItems(items, { continuation })` renders ordinary model input.
Before compaction, render the full prompt with system
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

A complete provider also implements native output/tool parsing and Responses
streaming. It calls `/v1/compact` and SGLang's native `/generate` on the same port;
the Codex base URL points to that provider. The example has been tested
against the live Remory backend; full interactive Codex sessions remain untested.
Check the Codex version's remote-compaction dispatch when adding the provider.

References: [Codex custom providers](https://developers.openai.com/codex/config-advanced/),
[Responses compaction](https://developers.openai.com/api/docs/guides/compaction),
[compaction source](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs).
