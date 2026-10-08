# Deployment

`./deploy.sh` installs the runtime and starts SGLang's native HTTP server with
the Remory model hook and compaction routes on the same port.
The default model is Qwen3.8-27B with the released Remory checkpoint. Both
backends read the context limit from the model configuration: 262,144 tokens
(256K) for Qwen3.8-27B.

## Requirements

- Linux x86_64, Python 3.11+ with `venv`, `git`, and a C++ compiler.
- One NVIDIA GPU. Budget at least 80 GB of GPU memory and 100 GB of disk.
- A driver supporting CUDA 12.8, or CUDA 13.0 for SM 10.3+ GPUs.

The launcher chooses the CUDA build and installs compiler components locally.
The first run downloads packages and weights, then compiles kernels; later runs
reuse the installation. Downloads use GitHub, PyPI, PyTorch, NVIDIA, and Hugging Face.

The full model has been tested on an NVIDIA L20D with CUDA 13.0. The 80 GB figure
is a capacity recommendation; memory use depends on context length and GPU type.
The current SGLang backend handles text requests serially on one GPU.

## Options

```bash
# Choose a GPU.
CUDA_VISIBLE_DEVICES=1 ./deploy.sh

# Optionally use a smaller context window to save GPU memory.
./deploy.sh --context-limit 32768

# Use another disk for the runtime and session database.
./deploy.sh --runtime-dir /path/to/remory-runtime

# Load local snapshots of the released models.
./deploy.sh --actor /path/to/Qwen3.8-27B --checkpoint /path/to/Remory-Qwen3.8-27B

# Install without starting the server.
./deploy.sh --install-only
```

| Option | Default | Description |
| --- | --- | --- |
| `--context-limit` | Model maximum | Input and generated tokens combined |
| `--gpu` | `0` | GPU index within `CUDA_VISIBLE_DEVICES` |
| `--host` | `127.0.0.1` | API bind address |
| `--port` | `8421` | API port |
| `--runtime-dir` | `.remory` | Environment, kernel cache, and session data |
| `--cuda` | `auto` | CUDA build: `auto`, `cu128`, or `cu130` |

Requests that exceed the context limit return an error. Reduce `--context-limit`
if the model runs out of GPU memory. If a repository update changes the pinned
runtime recipe, install it in a new runtime directory.

## Session data

The SQLite database at `.remory/memory.sqlite` stores summaries and residual
tensors across restarts; keep it with your session data. Model weights use the
Hugging Face cache.

When using a custom runtime directory or local weights, run the example with
that environment and those model paths:

```bash
/path/to/remory-runtime/venv/bin/python examples/quickstart.py \
    --actor /path/to/Qwen3.8-27B \
    --checkpoint /path/to/Remory-Qwen3.8-27B
```

## Transformers

For direct model inspection, a Transformers backend is also available:

```bash
python -m pip install -e '.[server,transformers]'
remory serve --backend transformers --device cuda:0 --store remory.sqlite
```

Both backends use the same compaction engine and verify checkpoint checksums.
See [development](development.md) for the SGLang hooks and backend interface.
