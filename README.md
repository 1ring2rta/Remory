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

### 1. Deploy SGLang

```bash
git clone https://github.com/1ring2rta/Remory.git
cd Remory
./deploy.sh
```

The launcher installs SGLang with the Remory hook and loads the actor and memory
network in the same worker. It serves SGLang's native API at
`http://127.0.0.1:8421`, using the model's full context window (256K for Qwen3.8-27B).
The first run downloads weights and compiles GPU kernels. To choose a GPU, use
`CUDA_VISIBLE_DEVICES=1 ./deploy.sh`.

### 2. Generate

```bash
curl http://127.0.0.1:8421/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is", "sampling_params":{"temperature":0,"max_new_tokens":16}}'
```

### 3. Compact and continue

In another terminal, run the example from the repository root:

```bash
.remory/venv/bin/python examples/quickstart.py
```

The example generates an answer, creates a summary and residual memory, then
continues from that checkpoint. All model calls use the same SGLang `/generate`
endpoint. During residual encoding, the Remory hook reads the actor's hidden
states and runs the memory network. During continuation, it inserts the saved
soft tokens after the summary.

## Replace compact

[`Harness.compact`](src/remory/adapters/harness.py) is the compaction method to
connect to your agent: generate a summary, encode residual memory, then return
the new checkpoint. The agent retains its own tools and compaction trigger.

```python
from remory.adapters import Harness
from remory.client import Client

with Client("http://127.0.0.1:8421", session_id="my-session") as client:
    agent = Harness(client, codec)  # The actor's native chat template
    checkpoint = agent.compact(
        prefix_messages=prefix,    # System prompt, tools, and original task
        removed_messages=history,
        summary_prompt=summary_prompt,
        summary_schema=summary_schema,
    )
    result = agent.generate(
        checkpoint=checkpoint, prefix_messages=prefix, messages=new_messages,
    )
```

The [complete example](examples/quickstart.py) loads the tokenizer and summary
prompt from the released checkpoint. If your harness already generates a
summary, use `on_compact(..., summary=summary)` to add residual memory.

Save the checkpoint with the conversation. For the next compaction, pass it as
`previous` and supply the newly removed history. Memory is stored in
`.remory/memory.sqlite`. The [Codex example](integrations/codex) shows where to
override a Responses provider's compact handler and resume generation.
See the [API reference](docs/api.md) for native SGLang requests with memory.

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
