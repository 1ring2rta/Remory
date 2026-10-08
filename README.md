# Remory: Learning Residual Memory for Context Compaction

**Remory** supplements an agent's compaction summary with learned soft memory
from its history. The memory is conditioned on the summary and appended after
it, helping the frozen model continue its work across compactions.

[**Paper**](https://huggingface.co/mocoV3/Remory-Qwen3.8-27B/blob/main/paper/remory.pdf) |
[**Models**](https://huggingface.co/mocoV3/Remory-Qwen3.8-27B)

![Remory: summary-conditioned residual memory alongside standard Codex compaction, with training curves from the paper.](assets/remory.png)

## Supported Models

| Base model | Memory checkpoint | Backends |
| --- | --- | --- |
| Qwen3.8-27B | [Remory-Qwen3.8-27B](https://huggingface.co/mocoV3/Remory-Qwen3.8-27B) | SGLang, Transformers |

## Quick Start

Requires Linux, Python 3.11+, `git`, a C++ compiler, and an NVIDIA GPU.
We recommend 80 GB or more GPU memory. See [deployment](docs/deployment.md)
for driver requirements and local model paths.

```bash
git clone https://github.com/1ring2rta/Remory.git
cd Remory
./deploy.sh
```

The launcher installs SGLang, downloads the model weights, and starts the service
at `http://127.0.0.1:8421`, using the model's full context window (256K for
Qwen3.8-27B). The first run also compiles GPU kernels.

In another terminal, run the example from the repository root:

```bash
.remory/venv/bin/python examples/quickstart.py
```

It compacts a short conversation and continues from the resulting memory.
To select a GPU:

```bash
CUDA_VISIBLE_DEVICES=1 ./deploy.sh
```

## Usage

Call `compact` when the agent creates a new summary, then pass the returned handle
to `generate` on subsequent model calls. The harness supplies native token IDs
and continues to execute tools.

```python
from pathlib import Path
from remory.client import Client

with Client("http://127.0.0.1:8421", session_id="my-session",
            api_key=Path(".remory/api-key").read_text().strip()) as client:
    memory = client.compact(
        prefix_ids=prefix_ids,    # System prompt, tools, and original task
        history_ids=history_ids,  # History being replaced
        summary_ids=summary_ids,
    )
    result = client.generate(
        handle=memory["handle"],
        continuation_ids=continuation_ids,
        max_new_tokens=1024,
    )
```

Save the handle with the summary. For the next compaction, pass it as `previous`
and supply the newly removed history. Session state is stored in
`.remory/memory.sqlite`.

The [Codex example](integrations/codex) connects these calls to a custom Responses
provider. The provider handles prompt rendering, tool parsing, and streaming;
the included bridge handles residual compaction and recovery.
See the [API reference](docs/api.md) for request fields and session handling.

## Results

![BrowseComp and Terminal-Bench 2.1 scores for Qwen3.8-27B and GLM-5.3-Flash, with and without residual memory.](assets/benchmarks.png)

Qwen3.8-27B and GLM-5.3-Flash improve across long-horizon agent benchmarks.
Residual memory also reduces repeated tool outputs and tool errors on BrowseComp
and Terminal-Bench 2.1. On SummHay, it approaches the full-context joint score
using 5.2% of the input positions.

Gray bars show published frontier results under their respective evaluation
protocols. Full results and settings are in the [paper](https://huggingface.co/mocoV3/Remory-Qwen3.8-27B/blob/main/paper/remory.pdf).

## Acknowledgements

The Qwen memory network builds on [DFlash](https://github.com/z-lab/dflash)
and [SpecForge](https://github.com/sgl-project/SpecForge).
Inference uses [SGLang](https://github.com/sgl-project/sglang)
and [Transformers](https://github.com/huggingface/transformers).

## Citation

```bibtex
@misc{xia2026remory,
  title  = {Remory: Learning Residual Memory for Context Compaction},
  author = {Xia, Hanchen and Chen, Baoyou and Ge, Yutang and Deng, Naihao and
            Yang, Senqiao and Dong, Zilong and Yuan, Weihao and Zhu, Siyu},
  year   = {2026},
  url    = {https://github.com/1ring2rta/Remory}
}
```

[Development](docs/development.md) · [MIT License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
