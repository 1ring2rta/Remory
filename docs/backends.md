# Deployment

Start with `./deploy.sh` from the repository root. The script owns the complete
SGLang setup and starts one Remory endpoint; there is no second service to configure.

## Common options

```bash
# Select a GPU and a smaller context.
CUDA_VISIBLE_DEVICES=1 ./deploy.sh --context-limit 8192

# Put the environment and session state on a larger disk.
./deploy.sh --runtime-dir /path/to/remory-runtime

# Reuse local snapshots of the released weights.
./deploy.sh --actor /path/to/Qwen3.8-27B --checkpoint /path/to/Remory-Qwen3.8-27B

# Install dependencies without loading the models.
./deploy.sh --install-only
```

The default context ceiling is 32768 tokens, including generation. Overflow fails
before inference instead of truncating history. Context capacity also depends on
GPU memory. `--gpu` selects a device within `CUDA_VISIBLE_DEVICES`; `--port` and
`--host` change the API address. `--cuda cu128` / `--cuda cu130` override automatic
CUDA selection when supported by the GPU and driver.

The API key is generated in `.remory/api-key` with owner-only permissions. Set
`REMORY_API_KEY` to supply your own key. Clients also send a stable session ID.
The database is `.remory/memory.sqlite`; retain it with session backups. When using
`--runtime-dir`, pass that same path to `examples/quickstart.py`.

First installation requires internet access to GitHub, PyPI, PyTorch, NVIDIA, and
Hugging Face. Python needs its `venv` module; kernel compilation needs a C++
compiler. The launcher installs CUDA components privately and leaves the system
CUDA installation unchanged. Downloads and compilation can take several minutes.
A changed deployment recipe requires a new runtime directory; keep the existing
database if you need to migrate session state.

## What SGLang runs

SGLang is pinned to commit `f08726fd56c7ff6d8bd258f1545f98148fa4ef58`
(version 0.5.13). [The patch](../deploy/sglang.patch) adds two small connections:
model-hook initialization and complete return of packed memory rows. The installer
checks source hashes before patching; the worker verifies the patched files at
startup. [The recipe](../src/remory/backends/sglang_recipe.json) is authoritative.

The Remory process starts and stops its own worker. That worker loads the actor
and compressor, captures selected decoder states during a complete causal prefill,
and injects residual embeddings before subsequent prefills. A partial source block
still produces its full memory allocation; padding never becomes source history.

This release uses one GPU and serializes requests. Shared prefix caching, chunked
prefill, CUDA graphs, and overlapping schedules are disabled so different residual
memories cannot reuse one another's states. Normal autoregressive KV/recurrent
caching within each generation remains enabled. Tensor parallelism and multimodal
input are outside this deployment's supported scope.

Hardware validation uses an NVIDIA L20D with CUDA 13.0, the full Qwen3.8-27B actor,
and released Remory compressor. The 80 GB recommendation is a starting capacity,
not a measured guarantee for every GPU or context size. See [validation](validation.md).

## Transformers reference backend

For inspection or another supported device, the repository also contains a direct
Transformers implementation:

```bash
python -m pip install -e '.[server,transformers]'
export REMORY_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
remory serve --backend transformers --device cuda:0 --store remory.sqlite
```

It captures the same selected layers and injects memory through `inputs_embeds`.
Both paths load frozen models and verify the compressor's configuration and weight
checksums. They share the compaction engine and state contract.

To add an inference engine, implement `Backend.encode` and `Backend.generate` in
[`remory.types`](../src/remory/types.py). The engine owns recursive compaction and
state; a backend supplies source/summary encoding and generation with embeddings.
