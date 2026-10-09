# Third-party notices

`src/remory/models/layers.py`, `src/remory/models/compressor.py`, and
`src/remory/pyramid.py` are inference-focused adaptations of the Recursive Memory
implementation (MIT, copyright 2026 Recursive Memory contributors). The loader
targets the public `mocoV3/Qwen3.8-27B-REMORY-1.9B` release. Training-loss code, target
training wrappers, training launchers, and evaluation harnesses were excluded.

The compressor architecture incorporates work from
[SpecForge](https://github.com/sgl-project/SpecForge) (MIT; upstream provenance
commit `204299a02fa31e56b5d425661107ae6cf247e4c6`) and DFlash (MIT).
Their notices are retained in `LICENSES/SpecForge-MIT.txt` and
`LICENSES/DFlash-MIT.txt`. The package imports, rather than vendors, Transformers
and PyTorch. Deployment fetches a pinned SGLang checkout (Apache-2.0) and applies the small
patch in `deploy/sglang.patch`. Its source revision and exact patch contents are
recorded in `src/remory/backends/sglang_recipe.json`; the Apache-2.0 license is
included in `LICENSES/SGLang-Apache-2.0.txt`. CUDA components are downloaded from
NVIDIA with checksum verification and retain their bundled license files. Model
weight licenses are separate from this code license.

Codex is an independent upstream project. The integration example does not imply
upstream endorsement or full client certification.
