# Third-party notices

`src/remory/models/layers.py`, `src/remory/models/compressor.py`, and
`src/remory/pyramid.py` are inference-focused adaptations of the Recursive Memory
implementation (MIT, copyright 2026 Recursive Memory contributors). The loader
targets the public `mocoV3/Remory-Qwen3.8-27B` release. Training-loss code, target
training wrappers, training launchers, and evaluation harnesses were excluded.

The compressor architecture incorporates work from
[SpecForge](https://github.com/sgl-project/SpecForge) (MIT; upstream provenance
commit `204299a02fa31e56b5d425661107ae6cf247e4c6`) and DFlash (MIT).
Their notices are retained in `LICENSES/SpecForge-MIT.txt` and
`LICENSES/DFlash-MIT.txt`. The package imports, rather than vendors, Transformers
and PyTorch. The SGLang adapter implements a wire protocol and includes no SGLang
server source. Model weight licenses are separate from this code license.

Codex, Claude Code, OpenCode, and Pi are names of independent upstream projects.
Integration examples do not imply upstream endorsement or full client certification.
