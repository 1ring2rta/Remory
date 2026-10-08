# Development

```bash
python -m pip install -e '.[test]'
pytest -q
node --test integrations/test.mjs
```

Install `.[test,transformers]` to include the tiny Qwen model and memory-hook tests.
These run locally without downloading the released 27B model. An optional GitHub
Actions configuration is in [ci-tests.yml](ci-tests.yml).

## Inference code

[`engine.py`](../src/remory/engine.py) implements compaction and generation.
[`pyramid.py`](../src/remory/pyramid.py) keeps memory within the slot budget, and
[`store.py`](../src/remory/store.py) persists immutable checkpoints in SQLite.

[`sglang_server.py`](../src/remory/sglang_server.py) extends SGLang's native HTTP
server. Normal answers and summaries go through `/generate`; residual encoding
also calls `/generate`, with `max_new_tokens=0` and Remory parameters. Its
[model hook](../src/remory/backends/sglang_hook.py) captures selected decoder states
and replaces placeholder embeddings with residual memory during prefill. The
worker loads both the actor and memory network. It uses full prefills and
serializes requests. Shared prefix caching, CUDA graphs, and overlapping
schedules are disabled; generation keeps its own KV and recurrent state.

SGLang is pinned to `f08726fd56c7ff6d8bd258f1545f98148fa4ef58` (0.5.13).
[Two patches](../deploy/sglang.patch) initialize the hook and return packed memory
rows, including partial source blocks. The installer and worker verify their
hashes against the [recipe](../src/remory/backends/sglang_recipe.json).

`Harness.compact` generates the summary, then calls `/v1/compact` to build and
save residual memory. That route uses the same server's native `/generate`
endpoint for each encoding pass. A subsequent `/generate` request with a
`remory.handle` restores the summary and embeddings before SGLang schedules it.

For another inference engine, implement `Backend.encode` and `Backend.generate`
in [`types.py`](../src/remory/types.py). The shared engine handles compaction,
recursion, and storage; the backend supplies encoding and embedding-aware generation.

## Tested setup

The 2026-10-07 check used Python 3.12, Node 22, one NVIDIA L20D, CUDA 13.0,
and the full released Qwen3.8-27B and Remory weights. A clean `deploy.sh`
installation passed the Python quickstart and JavaScript Codex example.

Live requests covered generation before compaction, two consecutive compactions,
partial source blocks, and recovery through a new client. The automated tests
also cover database reopen, session scoping, source-state capture, and failed
compactions. On 2026-10-08, the native SGLang deployment passed the Python
quickstart with a generated summary, the Codex example, two successive compactions,
resume through a new client, and native SSE with residual memory. It used the
model's default 262,144-token context window. Smoke inputs were short; this was
not a full-length evaluation. CUDA 12.8 and other GPUs have not been tested locally.
