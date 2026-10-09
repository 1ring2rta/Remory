# Assets

## Paper figures

- `remory.png`: the residual-compensation figure from the paper, rendered from
  `figures/iclr_residual_compensation.pdf` at 216 dpi. All four panels are retained.
- `benchmarks.png`: unchanged copy of `figures/headline_benchmarks.png` from the
  paper. GLM BrowseComp scores are 84.9 without and 89.0 with residual memory.

Source: [Remory preprint](https://huggingface.co/mocoV3/Qwen3.8-27B-REMORY-1.9B/tree/main/paper).

## Demo

The 54-second video uses the paper's method diagram and recorded task traces.
It moves from a paired Core War replay to four tasks: Core War, neural-network
parameter recovery, cricket tournament identification, and writer-background
research.
Each task shows both runs side by side, with matching context-chart axes and
separate repeat, error, and compaction counts. These examples were selected for
baseline failure and residual success, with compaction in both runs. Per-case
tool counts are shown as recorded; the aggregate charts use the full benchmark
results from the paper. Final outcomes and chart reveals are edited for
presentation, with no claim of synchronized execution or isolated causality.

Music: [Electric Dreams](https://www.scottbuckley.com.au/library/electric-dreams/)
by [Scott Buckley](https://www.scottbuckley.com.au), released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The excerpt is trimmed to 53.93 seconds, faded in and out, and level-adjusted.
