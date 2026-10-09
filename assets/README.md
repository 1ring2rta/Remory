# Assets

## Paper figures

- `remory.png`: the residual-compensation figure from the paper, rendered from
  `figures/iclr_residual_compensation.pdf` at 216 dpi. All four panels are retained.
- `benchmarks.png`: unchanged copy of `figures/headline_benchmarks.png` from the
  paper. GLM BrowseComp scores are 84.9 without and 89.0 with residual memory.

Source: [Remory preprint](https://huggingface.co/mocoV3/Qwen3.8-27B-REMORY-1.9B/tree/main/paper).

## Demo

The 54-second video uses the paper's method diagram and recorded task traces.
It moves from a paired Core War replay to four tasks: Core War, cell
segmentation, a Scheme evaluator, and a BrowseComp search.
Each task shows both runs side by side, with matching context-chart axes and
separate repeat, error, and compaction counts. Both cell runs, both Scheme runs,
and both retrieval runs pass; the cell baseline finishes without compaction.

The retrieval view shows the 30-frame contact sheet returned to GLM by its image
tool, with enlarged frames from the same sheet. It comes from
[candidate performance footage](https://archive.org/download/youtube-HAXtcuXxiiY/HAXtcuXxiiY.mp4)
examined during the search, and does not depict the final identified singer.
The overview shows recorded artifacts and final outcomes; chart reveals are
edited for presentation. No performance audio is used.

Music: [Electric Dreams](https://www.scottbuckley.com.au/library/electric-dreams/)
by [Scott Buckley](https://www.scottbuckley.com.au), released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The excerpt is trimmed to 53.93 seconds, faded in and out, and level-adjusted.
