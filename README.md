# Remory

**Compaction with residual memory, at the inference and harness boundary.**

[Weights](https://huggingface.co/mocoV3/Remory-Qwen3.8-27B) ·
[Paper](https://huggingface.co/mocoV3/Remory-Qwen3.8-27B/blob/main/paper/remory.pdf) ·
[API](docs/api.md) · [Harness integration](docs/harnesses.md)

An agent keeps its tools, conversation format, and compaction policy. Remory keeps
the information left behind by its text summary as continuous residual memory,
then injects that memory into the actor's input embeddings on later requests.

This repository contains the inference SDK, compressor loader, backend adapters,
an HTTP sidecar, and small harness integration modules. It contains no training
runner, dataset pipeline, or benchmark harness.

```text
your harness                        Remory                         actor
native history + new summary ─────► compact ─────► residual encoding
save summary + memory handle ◄───── checkpoint
handle + native continuation ─────► generate ────► embedding injection
execute returned tool calls ◄────── generated tokens
```

## Two integration points

| Interface | When to call it | Result |
| --- | --- | --- |
| `compact` | Before replacing old history with a summary | Durable, immutable memory handle |
| `generate` | To continue from a saved memory handle | Generated text, token IDs, and usage |

`generate` restores the saved memory and assembles the actor's input internally.

On repeated compactions, the old summary and old residual are re-encoded together
with the new history, conditioned on the **new** summary. A failed operation does
not publish a checkpoint or modify an existing one. History is never silently
truncated. The caller commits its transcript change only after `compact` succeeds.

## Install

Python 3.11 or later:

```bash
git clone https://github.com/1ring2rta/Remory.git
cd Remory
pip install -e '.[server,transformers]'
```

The Transformers reference backend loads the released Qwen actor and residual
compressor directly. It requires enough device memory for both models and the
requested context; it is an unbatched reference implementation. The default actor
is **Qwen3.8-27B**, not the model normally bundled with your coding client.

```bash
export REMORY_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
remory serve --backend transformers --device cuda:0 --store ./remory.sqlite
```

The loader pins the compressor release and actor revision and verifies compressor
checksums. It downloads safetensors and metadata, without executing code from the
model repository. Use `--checkpoint /path/to/snapshot --actor /path/to/actor` for
local snapshots of those same weights.

For an existing residual-enabled SGLang worker, install `.[server]` and see
[the SGLang instructions](docs/backends.md). The client checks capabilities,
checkpoint identity, context limits, and cache-bypass acknowledgments. The
required worker extensions are not part of stock SGLang.

## Use it from a harness

```python
from remory.client import Client

with Client("http://127.0.0.1:8421", session_id="my-session", api_key=api_key) as memory:
    checkpoint = memory.compact(
        prefix_ids=native_system_tools_and_original_task_ids,
        history_ids=removed_history_ids,
        summary_ids=new_summary_ids,
    )
    # Persist checkpoint["handle"] with the summary, then replace old history.
    result = memory.generate(
        handle=checkpoint["handle"],
        continuation_ids=kept_tail_and_next_request_ids,
        max_new_tokens=1024,
    )
```

These are **actor-native token IDs**. Retain the original tool definitions,
tool-call IDs, results, and chat-template boundaries. `Harness` and `ChatTemplate`
in [`remory.adapters`](src/remory/adapters/harness.py) provide a message-level
wrapper for a tokenizer's native template. See [the executable client example](examples/client.py).

For another inference engine, implement the two operations in
[`Backend`](src/remory/types.py): encode source/summary states, and generate with
embedding overrides. Remory assembles the token IDs and sparse memory overrides
before calling the backend.

## Harness support

| Harness | Included seam | Integration required |
| --- | --- | --- |
| Codex | [Responses compaction carrier](integrations/codex/bridge.mjs) | Wire into a custom Responses provider's compact and generation paths |
| Claude Code | [Compaction lifecycle bridge](integrations/claude-code/bridge.mjs) | Capture native source in a residual-aware gateway; persist PostCompact result there |
| `opencode-ai/opencode` | [Go client](integrations/opencode/remory.go) | Connect `agent.Summarize` and the model provider; persist handles across the new continuation session |
| Pi | [`session_before_compact` extension factory](integrations/pi/extension.mjs) | Supply native rendering/summary callback and a residual model provider |

These are small integration modules, **not four complete replacement providers**.
The sidecar implements `/v1/compact` and `/v1/generate`; it does not
implement the full OpenAI Responses or Anthropic Messages APIs. A client base-URL
setting alone is therefore insufficient. [The integration guide](docs/harnesses.md)
identifies the required hooks and the state each provider must carry.

Residual memory requires an actor that accepts hidden-state/embedding injection.
Closed Claude/GPT APIs cannot consume these tensors. Using Claude Code or Codex
as the harness with a supported Qwen backend is a separate choice from using
Anthropic's or OpenAI's hosted model.

## Checks

```bash
pip install -e '.[test]'
pytest -q
# Also runs a real tiny Qwen + residual-compressor integration when torch and
# transformers are installed. No large weight download is needed for tests.
node --test integrations/test.mjs
cd integrations/opencode && go test ./...
```

Tests cover recursive compaction, exact source coverage, sparse memory insertion,
failure atomicity, restart/resume, session scoping, and backend protocol contracts.
The individual coding CLIs have not been end-to-end certified with this release.
An optional GitHub Actions workflow is provided as [a template](docs/ci-tests.yml);
copy it to `.github/workflows/tests.yml` with workflow-authorized credentials to enable CI.

The current ABI is text-only. The sidecar stores float32 tensors in SQLite and
transports them as JSON to external backends; this favors portability over
throughput. It is designed for one trusted operator's sessions, protected by one
API key. Keep the database while resuming or branching sessions, and delete
unneeded handles explicitly. See [state and API semantics](docs/api.md).

## Attribution

*Remory: Learning Residual Memory for Context Compaction.* Hanchen Xia, Baoyou
Chen, Yutang Ge, Naihao Deng, Senqiao Yang, Zilong Dong, Weihao Yuan, and Siyu Zhu.

Code is MIT licensed. Model weight terms remain those of their respective model
repositories. See [third-party notices](THIRD_PARTY_NOTICES.md).
