# Development

```bash
python -m pip install -e '.[test]'
pytest -q
node --test integrations/test.mjs
```

Install `.[test,transformers]` to include the model and memory-hook tests.
These run locally without downloading the released 27B model. An optional GitHub
Actions configuration is in [ci-tests.yml](ci-tests.yml).

## Inference code

[`engine.py`](../src/remory/engine.py) implements compaction and generation.
[`pyramid.py`](../src/remory/pyramid.py) keeps memory within the slot budget, and
[`store.py`](../src/remory/store.py) persists immutable checkpoints in SQLite.

[`launch_server.py`](../src/remory/launch_server.py) parses launch options and
configures the installed CUDA toolchain before importing SGLang.
[`sglang_server.py`](../src/remory/sglang_server.py) extends SGLang's native HTTP
server. Normal answers, summaries, and residual encoding use `/generate`.
The Qwen [hook](../src/remory/backends/sglang_hook.py) captures decoder outputs
with a complete prefill and returns packed memory rows. It disables shared
prefix caching and uses one active request.
The GLM [hook](../src/remory/backends/glm_hook.py) captures mHC attention inputs
across prefill chunks, scatters memory at absolute input positions, and confirms
consumption after the complete prefill. Its internal binary cache is separate
from the durable, session-bound SQLite store. Each request has a unique cache
salt. Both paths disable CUDA graphs and overlapping schedules.

SGLang is pinned to `575759d90af942eeff89f1c8c33a1fcdb2da4181` for both models.
The [patch](../deploy/sglang.patch) initializes the Qwen hook, returns packed
memory rows, and connects the GLM adapter and native image processor. The installer
and launcher verify all eight files against the [recipe](../src/remory/backends/sglang_recipe.json).

`Harness.compact` generates the summary, then calls `/v1/compact` to build and
save residual memory. That route uses the same server's native `/generate`
endpoint for each encoding pass. A subsequent `/generate` request with a
`remory.handle` restores the summary and embeddings before SGLang schedules it.

For another inference engine, implement `Backend.encode` and `Backend.generate`
in [`types.py`](../src/remory/types.py). The shared engine handles compaction,
recursion, and storage; the backend supplies encoding and embedding-aware generation.

## Tested setup

On 2026-10-09, the shared SGLang revision passed 42 Python tests and four
JavaScript tests, using Python 3.12, PyTorch 2.13.0, Transformers 5.12.1, CUDA 13.0,
and Node 22. All eight patched SGLang files passed checksum verification. The
installer's dependency resolution passed; a fresh environment installation has
not been rerun for this revision.

The released Qwen actor and memory weights passed the native launcher and Python
quickstart on two L20D GPUs, with the default 256K context configuration. Live
checks covered summary generation, a 53-token partial block returning all 64
soft tokens, two successive compactions, continuation through a new client,
native SSE, and the JavaScript Codex example. Inputs were short; this was not a
full-context evaluation. The 2 × 80 GB budget has not been measured on 80 GB cards.

The released 1.24B GLM encoder produced bitwise-identical CUDA outputs to the
local evaluation encoder for 37-token and 1,024-token source blocks. Eight
reference tests also passed for mHC feature capture, chunked memory injection,
request alignment, and vision-prefix validation. The full 320B GLM actor has
not been rerun through the public launcher; the integration ports the previously
evaluated local runtime on this same SGLang revision. The GPU checks used an
existing compatible environment and test-only settings for the shared node's
NCCL and available memory.
