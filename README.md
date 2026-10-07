# Remory

**Compaction with residual memory.** Keep a text summary, encode soft memory from
the history it replaces, and attach that memory when the agent continues.

[Paper](https://huggingface.co/mocoV3/Remory-Qwen3.8-27B/blob/main/paper/remory.pdf) ·
[Weights](https://huggingface.co/mocoV3/Remory-Qwen3.8-27B) ·
[API](docs/api.md) · [Codex example](docs/harnesses.md)

## 1. Deploy

On a Linux x86_64 GPU machine with Python 3.11+, `git`, and a C++ compiler:

```bash
git clone https://github.com/1ring2rta/Remory.git
cd Remory
./deploy.sh
```

This installs a private environment, a pinned SGLang runtime with the two required
patches, and CUDA compiler components. It downloads the released Qwen3.8-27B actor
and Remory weights, then serves **http://127.0.0.1:8421**. First startup includes
large downloads and kernel compilation; later starts reuse them. No other project
checkout or preconfigured SGLang server is needed.

Use one NVIDIA GPU with enough memory for the 27B actor, 1.94B compressor, and
context. Budget at least 80 GB of GPU memory and 100 GB of disk for the initial
installation and weights. The driver must support CUDA 12.8, or CUDA 13.0 for
SM 10.3+ GPUs. This release serves text on one GPU, one request at a time.
See [deployment options and tested hardware](docs/backends.md).

Runtime files live in `.remory/`: the environment, API key, and persistent memory
database. Keep this directory to resume sessions. Model downloads use the usual
Hugging Face cache. Stop the service with Ctrl-C.

## 2. Run a complete example

In a second terminal, from the repository root:

```bash
.remory/venv/bin/python examples/quickstart.py
```

The example supplies a short conversation and a summary, compresses the removed
history, then asks the actor to continue with the saved residual memory.
It prints the memory size and the actor's answer. No input file is needed.

## 3. Connect your harness

There are two public operations:

| Operation | Input | Output |
| --- | --- | --- |
| `compact` | Native history tokens + new summary | Persistent memory handle |
| `generate` | Memory handle + native continuation tokens | Actor output |

```text
history + summary ── compact ──► summary + residual memory
                                       │
new messages ───────────────── generate ──► answer / tool calls
```

The harness creates the summary and executes tools. Remory owns memory encoding,
storage, and injection into the actor. Save the returned handle with the summary
before discarding history. On the next compaction, supply that handle as `previous`.
The same two operations handle repeated compactions and session resume.

[The Codex example](docs/harnesses.md) shows where to connect both operations in a
custom Responses provider. It is the only harness-specific integration included.
The service exposes Remory's token API; a Codex base-URL change alone does not
supply the provider's rendering, tool parsing, and streaming logic. The example
uses the released Qwen actor; residual tensors require access to actor embeddings.

For your own client, see [the Python client example](examples/client.py) and
[the request formats](docs/api.md). Use the actor's native tokenizer and preserve
its tool-call IDs, results, and chat-template boundaries.

## Read the code

Follow the same path as a request:

| File | Purpose |
| --- | --- |
| [`deploy.sh`](deploy.sh) | Install and start the service |
| [`server.py`](src/remory/server.py) | The two HTTP operations |
| [`engine.py`](src/remory/engine.py) | Compact, save, restore, and generate |
| [`sglang.py`](src/remory/backends/sglang.py) | Send work to the managed SGLang process |
| [`sglang_hook.py`](src/remory/backends/sglang_hook.py) | Capture actor states and inject soft tokens |
| [`codex/bridge.mjs`](integrations/codex/bridge.mjs) | Carry memory across Codex compaction |

`models/` contains the released compressor; `pyramid.py` handles memory budgets;
`store.py` persists immutable checkpoints in SQLite. Training and evaluation
pipelines are outside this repository.

## Development

```bash
python -m pip install -e '.[test]'
pytest -q
node --test integrations/test.mjs
```

With `.[transformers]` installed, the tests also exercise tiny real Qwen models
and the SGLang hook. See [validation](docs/validation.md) for the tested scope and
[the optional CI template](docs/ci-tests.yml).

*Remory: Learning Residual Memory for Context Compaction.* Hanchen Xia, Baoyou
Chen, Yutang Ge, Naihao Deng, Senqiao Yang, Zilong Dong, Weihao Yuan, and Siyu Zhu.
Code: MIT. Model licenses are separate. [Third-party notices](THIRD_PARTY_NOTICES.md).
