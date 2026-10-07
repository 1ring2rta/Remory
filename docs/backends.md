# Inference backends

## Transformers reference implementation

Install `.[server,transformers]` and run the command in the README. The package
contains the inference modules needed to read the public compressor. It loads
the actor separately, captures decoder outputs at the checkpoint-selected layers,
and runs the residual compressor with the new summary's normalized final states.
Selected states come from a full causal prefill, not from independently encoded
source chunks. A partial leaf receives 64 slots without synthetic source tokens.

Generation scatters continuous rows into `inputs_embeds`. The actor and compressor
are frozen and run under `torch.inference_mode()`. Requests are serialized because
temporary capture hooks belong to one prefill. This reference path is useful for
correctness and provider integration; long-context and batched serving should use
an optimized implementation of the same `Backend` protocol.

The public 27B actor plus 1.94B compressor need substantial memory, beyond model
weights for activations/cache. `--context-limit` is an explicit request ceiling,
not a guarantee that this many tokens fit on a particular device. The default is
32768. Use the tested SGLang runtime for the larger evaluation context lengths.

## Existing residual-enabled SGLang worker

The lightweight client does not import torch or SGLang. Point it at a worker
already running the SummaryResidual hook and the intended checkpoint:

```bash
remory doctor --backend sglang \
  --backend-url http://127.0.0.1:30000 \
  --config /path/to/deployed-checkpoint/config.json \
  --server-checkpoint /worker/path/to/deployed-checkpoint \
  --server-model /worker/path/to/Qwen3.8-27B
```

Use the same arguments with `serve`, plus `--store`. Set `REMORY_BACKEND_API_KEY`
if the worker requires bearer authentication. Paths after `--server-*` are exact
strings in the worker's namespace; the client can run on another host. The config
must describe those exact immutable weights. A path check is not a remote weight
checksum; deployment operators must keep these paths bound to the intended files.

The worker must advertise all four capabilities:

- `qwen38_summary_residual_sglang_v1`
- `qwen38_recursive_summary_residual_v1`
- `qwen38_residual_dynamic_summary_v1`
- `request_cache_bypass_v1`

It must expose the `make_summary_residual_compressor_hook` factory, a safe batching
contract, and its input limit. Encode and recover requests require explicit
cache-bypass receipts. No global cache flushes are performed by this client.

This adapter targets the residual-serving ABI used by the original runtime in
[Recursive Memory](https://github.com/1ring2rta/recursive-memory). **A stock SGLang
installation is not sufficient.** This repository does not bundle or patch that
server; use the self-contained Transformers path if you do not already have a
compatible worker. `doctor` checks support rather than assuming it from a version
number. The original CPU-tensor JSON transport is preserved for compatibility.

## Bring another backend

Implement `Backend.encode` and `Backend.generate` from `remory.types`, exposing
the checkpoint's `Contract` and the engine's actual `context_limit`. Source
operations are `summary`, `leaves`, `recursive_leaves`, and `parent`. Recursive
source requests include sparse overrides for the previous soft memory. Parent
requests contain exactly one block of adjacent child embeddings. The engine
owns the chronological pyramid and immutable state, so adapters need no training
or benchmark machinery.
