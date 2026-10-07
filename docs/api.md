# API and state contract

Use `Authorization: Bearer <REMORY_API_KEY>` and `X-Remory-Session: <stable ID>`
on every request except `/health`. The shared key authorizes one trusted operator;
session IDs scope state but are not separate authentication principals. Default
binding is `127.0.0.1:8421`. FastAPI exposes an OpenAPI schema at `/openapi.json`.

## `POST /v1/compact`

```json
{
  "prefix_ids": [1, 2],
  "history_ids": [3, 4, 5],
  "summary_ids": [6, 7],
  "previous": null
}
```

The numbers above are illustrative, not a Qwen prompt. Supply IDs from the exact
actor tokenizer and native chat template. `prefix_ids` contains the immutable
system/tools/original-task prefix. `history_ids` is the span being removed. Keep
recent messages outside that span and append them during `prepare`/`generate`.
Do not cut across an outstanding tool call/result boundary.

Remory does not generate the summary in this endpoint. The harness supplies its
new summary. The released checkpoint uses a JSON handoff with five arrays:
`current_progress`, `key_decisions`, `important_context_constraints_preferences`,
`next_steps`, and `critical_data_examples_references`. Preserve that format when
reproducing the released model's behavior. Other summary policies are accepted by
the SDK but their quality has not been established here.

Response: `{ "handle": "rm_…", "summary_ids": [...], "receipt": {...} }`.
Only after receiving this response should the harness replace its old context.
Keep the handle in the same transaction/session entry as the text summary.

For the next compaction, set `previous` to the old handle, keep `prefix_ids`
identical, and pass **only newly removed raw history** in `history_ids`. Include
the previously retained tail if it is now being removed. Do not pass the old
summary or soft placeholders again: Remory reconstructs them. Each handle is
immutable, so retries and branches cannot overwrite another checkpoint. A retry
may allocate another handle; delete unused handles after the caller commits.

The receipt records source/summary hashes, source token count, memory size,
chronological pyramid spans, and logical depths. Checkpoint embedding depth
saturates at its configured maximum; logical depth remains recorded exactly.
An impossible budget raises an error rather than dropping source tokens.

## `POST /v1/prepare`

Input: `{ "handle": "rm_…", "continuation_ids": [...] }`.

Returns `input_ids`, `memory_positions`, `memory_embeddings`, and `slot_depths`.
The token stream is the saved native prefix, summary envelope, memory envelope,
then the exact supplied continuation. Scatter each memory row into the actor's
input embedding at the corresponding position **before** prefill. Placeholder
token IDs alone do not contain memory. Avoid ordinary token-prefix cache reuse
unless the backend's cache identity also includes the memory content.

Use `prepare` if your provider owns generation. Do not decode its placeholders
to text and send that text to an ordinary LLM API.

## `POST /v1/generate`

```json
{
  "handle": "rm_…",
  "continuation_ids": [8, 9],
  "max_new_tokens": 1024,
  "sampling": {"temperature": 0.0}
}
```

Remory restores and injects memory through its backend. The response contains
`text`, `output_ids`, `usage`, and `finish_reason`. It is not a Chat Completions
response. The harness/provider remains responsible for parsing native tool calls,
executing tools, and formatting their results. Before the first compaction, omit
`handle` and send full `input_ids` instead. Supplying both is an error.

## Lifecycle and failure handling

`DELETE /v1/memories/{handle}` deletes only a handle in the supplied session.
There is no automatic expiry. SQLite persists tensors without pickle and allows
resuming after a sidecar restart. Do not change weights, tokenizer, system prompt,
or tool definitions while reusing a checkpoint. Backend contract mismatches fail.
Treat the SQLite database as conversation data, and retain it with session backups.

400: invalid contract/input or context overflow. 401: invalid key. 404: no handle
for this session. 422: invalid JSON schema. 502: backend failure. In every failure
case, keep the original history and surface the error. Failed/cancelled clients
may leave unreferenced checkpoints if the server already completed; no active
checkpoint is overwritten. Backend generation is synchronous in this release.
