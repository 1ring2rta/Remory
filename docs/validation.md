# Initial release validation

Validated locally on 2026-10-07:

- 20 Python tests: engine, durable state, HTTP API, SGLang protocol, message
  template boundary, and real tiny Qwen3 and Qwen3.5-family hybrid actors with
  the residual compressor.
- 4 JavaScript tests: HTTP transport, Codex opaque compaction carrier, Pi
  cancellation/commit behavior, and Claude Code gateway lifecycle bridge.
- 2 Go tests: native source/continuation serialization and backend failure handling.
- Ruff checks, editable installation, CLI entry point, wheel build, and wheel
  license contents passed.
- The public Qwen compressor's 97 tensors loaded with exact equality to the
  published safetensors file. A real CPU forward with 32 source positions, four
  summary positions, and a 1024-position virtual slot allocation produced finite
  output of shape `[1, 64, 5120]`.

The tiny model tests compare selected source features and normalized summary
states with the actor's native outputs, performs two consecutive compactions,
check sparse embedding insertion, and generate continuation tokens. The hybrid
test uses the same Qwen3.5 multimodal model class as the released actor, with
small dimensions and CPU reference kernels; it exercises text input only.

No full 27B actor session was launched for this release validation. A live SGLang
worker and interactive Codex, Claude Code, OpenCode, and Pi sessions were not
available in this validation run. Their adapters have protocol/contract coverage,
not end-to-end client certification. No benchmark or quality improvement claim
is inferred from these implementation checks.
