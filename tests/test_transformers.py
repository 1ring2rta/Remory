# Optional heavy dependencies must be checked before importing the backend.
# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import Qwen3Config, Qwen3ForCausalLM
from remory import Contract, MemoryStore, Prepared, Remory
from remory.backends.transformers import TransformersBackend
from remory.models.compressor import SummaryResidualCompressor


class Decoder:
    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def test_real_tiny_actor_compressor_recursive_encode_and_generate(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=100, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=256, attention_dropout=0.0)
    actor = Qwen3ForCausalLM(config).eval()
    draft = Qwen3Config(vocab_size=100, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=256, layer_types=["sliding_attention", "full_attention"], sliding_window=8)
    compressor = SummaryResidualCompressor(draft, target_hidden_size=16, target_layer_ids=(0, 1),
        compression_ratio=2, local_window=2048, gradient_checkpointing=False, max_depth=4).eval()
    contract = Contract(16, 8, 2, 8, 4, 2, (10,), (11,), (12,), (13,), "tiny-qwen")
    backend = TransformersBackend(actor, compressor, Decoder(), contract, context_limit=200)
    source = Prepared((1, 4, 5, 6))
    observed = []
    hook = compressor.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().clone()))
    summary = backend.encode(source, start=2, end=4, operation="summary")
    backend.encode(source, start=1, end=4, operation="leaves", summary=summary)
    hook.remove()
    with torch.inference_mode():
        native = actor.model(input_ids=torch.tensor([source.input_ids]), output_hidden_states=True)
    np.testing.assert_allclose(summary, native.last_hidden_state[0, 2:4].numpy(), rtol=1e-5, atol=1e-6)
    expected = torch.cat([native.hidden_states[1][:, 1:4], native.hidden_states[2][:, 1:4]], -1)
    torch.testing.assert_close(observed[0], expected)
    engine = Remory(backend, MemoryStore(tmp_path / "tiny.sqlite"))
    first = engine.compact(owner="a", prefix_ids=[1], history_ids=list(range(20, 40)), summary_ids=[50, 51])
    second = engine.compact(owner="a", prefix_ids=[1], history_ids=[60, 61], summary_ids=[52], previous=first.handle)
    assert np.isfinite(second.embeddings).all()
    prepared = engine.prepare(owner="a", handle=second.handle, continuation_ids=[70])
    with torch.inference_mode():
        actual = backend._embed(prepared)[0, list(prepared.memory_positions)].numpy()
    np.testing.assert_array_equal(actual, second.embeddings)
    result = engine.generate(owner="a", handle=second.handle, continuation_ids=[70], max_new_tokens=2)
    assert 1 <= len(result.output_ids) <= 2
