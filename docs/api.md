# API

The service listens on `http://127.0.0.1:8421` by default. Send these headers:

```http
Authorization: Bearer <REMORY_API_KEY>
X-Remory-Session: <session-id>
```

The API key is shared by the server's clients; the session ID identifies their
stored memories. `GET /health` needs no key. The OpenAPI schema is at `/openapi.json`.

## Compact

`POST /v1/compact`

```json
{
  "prefix_ids": [1, 2],
  "history_ids": [3, 4, 5],
  "summary_ids": [6, 7],
  "previous": null
}
```

Use the actor's native tokenizer and chat template for all token fields.
`prefix_ids` contains the system prompt, tools, and original task;
`history_ids` contains the messages being removed. Keep complete tool call/result
pairs together. Retained messages go into the next generation's continuation.

The caller generates the summary. The released model uses a JSON summary with
five array fields:

```json
{
  "current_progress": [],
  "key_decisions": [],
  "important_context_constraints_preferences": [],
  "next_steps": [],
  "critical_data_examples_references": []
}
```

Response:

```json
{"handle": "rm_…", "summary_ids": [6, 7], "receipt": {"…": "…"}}
```

Save the handle with the summary before replacing the old history. `receipt`
records token counts, source hashes, memory size, and compression depths.

For subsequent compactions, set `previous` to the old handle, keep the prefix
identical, and send only newly removed history. Remory restores the previous
summary and memory before re-encoding them with that history. Each compaction
creates a new immutable handle, so branches can keep different checkpoints.

## Generate

`POST /v1/generate`

```json
{
  "handle": "rm_…",
  "continuation_ids": [8, 9],
  "max_new_tokens": 1024,
  "sampling": {"temperature": 0.0}
}
```

The service restores the saved prefix, summary, and soft memory, then appends
`continuation_ids`. The response contains `text`, `output_ids`, `usage`, and
`finish_reason`. The provider parses native tool calls and formats tool results.
Generation is synchronous.

Before the first compaction, omit `handle` and send the full prompt in `input_ids`.
Use either `input_ids` or a handle with `continuation_ids`.

## Stored memory

`DELETE /v1/memories/{handle}` deletes a checkpoint in the current session.
Handles have no automatic expiry. SQLite retains them across service restarts;
keep the database when resuming sessions. Reuse a handle with the same weights,
tokenizer, system prompt, and tool definitions.

A failed compaction leaves existing handles intact. Keep the original history
until a request succeeds. A cancelled request or retry can leave an unused handle
if the server has already finished; delete it after committing the chosen result.

| Status | Meaning |
| --- | --- |
| 400 | Invalid input, incompatible checkpoint, or context overflow |
| 401 | Invalid API key |
| 404 | Memory handle not found in this session |
| 422 | Invalid request schema |
| 502 | Inference backend failed |
