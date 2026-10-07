# Validation

Validated locally on 2026-10-07 with Python 3.12 and Node 22.

## Automated checks

- 22 Python tests cover compaction, recursive source coverage, failure atomicity,
  immutable SQLite state, session scoping, HTTP requests, and template boundaries.
- Tiny real Qwen and Qwen hybrid models verify selected source states, normalized
  summary states, embedding insertion, and consecutive compactions.
- The SGLang hook tests check fused residual reconstruction, direct `model.forward`
  dispatch, summary capture, injection, decode behavior, and cleanup after errors.
- Deployment tests verify that patches are idempotent and reject changed upstream
  source before writing either file.
- 3 JavaScript tests cover HTTP serialization and both Codex provider connections,
  including native input before compaction, checkpoint recovery, and rejection of
  foreign compaction items.

Ruff, wheel packaging (including the runtime recipe and licenses), and
`git diff --check` pass. The CI configuration is provided as a
[template](ci-tests.yml); it is not an enabled GitHub workflow.

## Full model on SGLang

`deploy.sh --install-only` completed in a new virtual environment, including a
fresh pinned SGLang checkout and private CUDA toolchain. `pip check` found no
broken requirements. The normal launcher then started this environment with the
default 32768-token context ceiling and local snapshots of the released weights;
the Python quickstart, repeated-compaction smoke run, and Codex example all passed.
The published weight revisions were also verified to be publicly accessible
without authentication. The smoke requests use short contexts, not the full ceiling.

The live smoke run uses one NVIDIA L20D, CUDA 13.0, pinned SGLang 0.5.13,
the released Qwen3.8-27B actor, and the public 1.94B Remory compressor.
It exercises the HTTP API with native Qwen tokens:

| Check | Result |
| --- | --- |
| Generation before compaction | Produces output tokens |
| Quickstart: 53 history tokens | 64 residual tokens, then successful continuation |
| 1113-token removed history | 128 residual tokens, including the partial final block |
| Second compaction with the previous handle | Re-encodes 128 prior soft tokens with new history; 64 new soft tokens |
| Four-token source request | Returns all 64 soft rows without truncating them to source length |
| Resume through a new client | Identical greedy output from the saved handle |
| Codex JavaScript example | Compacts, carries the item, restores memory, and generates |

The quickstart's continuation was `Publish the README.` State tests also cover
server-side database reopen and branch ownership. These are implementation checks;
no benchmark quality claim is inferred from the small sample.

The included Codex module has live backend coverage, but an interactive Codex
session with a complete custom Responses provider has not been certified.
The CUDA 12.8 recipe and other GPU architectures have not been exercised locally.
