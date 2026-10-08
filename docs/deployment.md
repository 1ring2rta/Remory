# Deployment

`python -m remory.launch_server` starts SGLang's native HTTP server with the
Remory model hook and compaction routes on the same port.
The default model is Qwen3.8-27B with the released Remory checkpoint. Both
backends read the context limit from the model configuration: 262,144 tokens
(256K) for Qwen3.8-27B.

## Requirements

- Linux x86_64, Python 3.11+ with `venv`, `git`, and a C++ compiler.
- One NVIDIA GPU. Budget at least 80 GB of GPU memory and 100 GB of disk.
- A driver supporting CUDA 12.8, or CUDA 13.0 for SM 10.3+ GPUs.

The installer chooses the CUDA build and installs compiler components locally.
The launcher downloads model weights and compiles kernels on first use.
Downloads use GitHub, PyPI, PyTorch, NVIDIA, and Hugging Face.

The full model has been tested on an NVIDIA L20D with CUDA 13.0. The 80 GB figure
is a capacity recommendation; memory use depends on context length and GPU type.
The current SGLang backend handles text requests serially on one GPU.

## Install

```bash
python deploy/install.py
source .remory/venv/bin/activate
```

The installer creates `.remory/venv` and installs the pinned SGLang source with
the Remory hooks. Run it once; starting the server does not install packages.
The launch module reads the installed CUDA and kernel-cache paths automatically.

To put the environment on another disk or choose a CUDA build:

```bash
python deploy/install.py --runtime-dir /path/to/remory-runtime --cuda cu130
source /path/to/remory-runtime/venv/bin/activate
```

`--cuda` accepts `auto` (the default), `cu128`, and `cu130`. If an update changes
the pinned runtime recipe, install it in a new runtime directory.

## Launch

```bash
python -m remory.launch_server \
    --model-path Qwen/Qwen3.8-27B \
    --remory-checkpoint mocoV3/Remory-Qwen3.8-27B \
    --host 127.0.0.1 --port 8421
```

`--model-path` accepts the checkpoint's target model ID or a local snapshot.
The published model ID resolves to the actor revision recorded in the checkpoint.
Both model arguments can be omitted to use the released Qwen defaults.

```bash
# Choose a GPU.
CUDA_VISIBLE_DEVICES=1 python -m remory.launch_server

# Optionally use a smaller context window to save GPU memory.
python -m remory.launch_server --context-length 32768

# Load local snapshots of the released models.
python -m remory.launch_server \
    --model-path /path/to/Qwen3.8-27B \
    --remory-checkpoint /path/to/Remory-Qwen3.8-27B
```

| Option | Default | Description |
| --- | --- | --- |
| `--context-length` | Model maximum | Input and generated tokens combined |
| `--mem-fraction-static` | `0.8` | SGLang static GPU memory fraction |
| `--base-gpu-id` | `0` | GPU index within `CUDA_VISIBLE_DEVICES` |
| `--host` | `127.0.0.1` | API bind address |
| `--port` | `8421` | API port |
| `--store` | `<runtime-dir>/memory.sqlite` | Session database |
| `--runtime-dir` | Detected from the environment | Installed CUDA paths, kernel cache, and session data |

Requests that exceed the context limit return an error. Reduce `--context-length`
if the model runs out of GPU memory. Run `python -m remory.launch_server --help`
for the supported options. The current integration serves one GPU per process.

The original `./deploy.sh` shortcut remains available and uses this same launcher.

## Session data

The SQLite database at `.remory/memory.sqlite` stores summaries and residual
tensors across restarts; keep it with your session data. Model weights use the
Hugging Face cache.

When using a custom runtime directory or local weights, run the example with
that environment and those model paths:

```bash
python examples/quickstart.py \
    --model-path /path/to/Qwen3.8-27B \
    --remory-checkpoint /path/to/Remory-Qwen3.8-27B
```

## Transformers

For direct model inspection, a Transformers backend is also available:

```bash
python -m pip install -e '.[server,transformers]'
remory serve --backend transformers --device cuda:0 --store remory.sqlite
```

Both backends use the same compaction engine and verify checkpoint checksums.
See [development](development.md) for the SGLang hooks and backend interface.
