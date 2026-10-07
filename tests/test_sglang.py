from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from remory import Contract, Prepared  # noqa: E402
from remory.backends.sglang import SGLangBackend  # noqa: E402
from remory.backends.sglang_hook import ResidualHook, validate_payload  # noqa: E402


def encode_payload(**overrides):
    return {"mode": "encode", "operation": "leaves", "start": 1, "end": 4,
            "depth": 0, "block_depths": [], "summary": [[1., 2., 3., 4.]], **overrides}


def test_source_and_recursive_depth_validation():
    validate_payload(encode_payload(), 4, 4, block=8)
    for override in ({"end": 3}, {"summary": [[float("nan")]*4]},
                     {"depth": True}, {"positions": [1], "memory": [[0.]*4]}):
        with pytest.raises(ValueError):
            validate_payload(encode_payload(**override), 4, 4, block=8)
    recursive = encode_payload(operation="recursive_leaves", positions=[1, 2],
                               memory=[[0.]*4]*2, block_depths=[1])
    validate_payload(recursive, 4, 4, block=8)
    with pytest.raises(ValueError, match="identify"):
        validate_payload({**recursive, "block_depths": [0]}, 4, 4, block=8)
    with pytest.raises(ValueError, match="exactly one block"):
        validate_payload(encode_payload(operation="parent", depth=1), 4, 4, block=8)


def test_backend_rejects_truncated_partial_leaf_and_preserves_overrides():
    backend = SGLangBackend.__new__(SGLangBackend)
    backend.contract = Contract(4, 8, 2, 8, 4, 2, (), (), (), (), "test")
    seen = []
    def request(source, sampling, payload):
        seen.append(payload)
        return {"meta_info": {"hidden_states": [[[1.]*4]*4]}}
    backend._request = request
    result = backend.encode(Prepared((1, 3, 4)), start=1, end=3, operation="leaves",
                            summary=np.ones((1, 4)))
    assert result.shape == (4, 4)  # More memory rows than the short source span.
    backend._request = lambda *args: {"meta_info": {"hidden_states": [[[1.]*4]*2]}}
    with pytest.raises(RuntimeError, match="incomplete"):
        backend.encode(Prepared((1, 3, 4)), start=1, end=3, operation="leaves",
                       summary=np.ones((1, 4)))
    def generate(source, sampling, payload):
        assert payload["positions"] == [1]
        assert payload["memory"] == [[5.]*4]
        assert sampling["stop_token_ids"] == [2]
        return {"text": "ok", "output_ids": [2], "meta_info": {"completion_tokens": 1}}
    backend._request = generate
    assert backend.generate(Prepared((1, 2, 3), (1,), np.full((1, 4), 5)),
                            max_new_tokens=1).text == "ok"


class FusedLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)

    def forward(self, hidden, residual):
        total = hidden if residual is None else hidden + residual
        return self.linear(total), total


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(20, 4)
        self.layers = torch.nn.ModuleList([FusedLayer() for _ in range(4)])
        self.norm = torch.nn.LayerNorm(4)

    def forward(self, ids):
        hidden, residual = self.embed_tokens(ids), None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual)
        return self.norm(hidden + residual)


class Logits(torch.nn.Module):
    def forward(self, ids, hidden, head, batch):
        return SimpleNamespace(hidden_states=hidden)


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.logits_processor = Backbone(), Logits()

    def forward(self, input_ids, positions, forward_batch):
        hidden = self.model(input_ids)
        return self.logits_processor(input_ids, hidden, None, forward_batch)


class Compressor(torch.nn.Module):
    target_layer_ids, max_depth = (0, 2), 4
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.observed = []

    def forward(self, features, mask, **kwargs):
        self.observed.append(features.clone())
        assert kwargs["slot_token_lengths"].tolist() == [8]
        rows = features[:, :, :4].mean(1, keepdim=True).expand(1, 4, 4).clone()
        return SimpleNamespace(embeddings=rows, attention_mask=torch.ones(1, 4, dtype=torch.bool))


def batch(payload, *, prefix=0, decode=False):
    return SimpleNamespace(sampling_info=SimpleNamespace(custom_params=[{"remory": payload}]),
        batch_size=1, is_prefill_only=payload["mode"] == "encode",
        forward_mode=SimpleNamespace(is_decode=lambda: decode, is_extend=lambda: not decode),
        extend_prefix_lens_cpu=[prefix], extend_seq_lens_cpu=[4])


def test_hook_matches_fused_decoder_states_and_restores_per_request_state(monkeypatch):
    torch.manual_seed(42)
    actor, compressor = Actor(), Compressor()
    config = {"compressor": {"target_hidden_size": 4},
              "residual": {"block_tokens": 8, "compression_ratio": 2}}
    monkeypatch.setattr("remory.models.load.load_compressor", lambda *a, **kw: (compressor, config))
    hook = ResidualHook({"checkpoint": "unused"})
    hook.setup_model(actor)
    actor.logits_processor.register_forward_hook(hook)
    ids = torch.tensor([1, 2, 3, 4])
    # Independent decoder-state reconstruction, including each residual add.
    total = actor.model.embed_tokens(ids)
    states = []
    for layer in actor.model.layers:
        total = layer.linear(total) + total
        states.append(total)
    # SGLang invokes forward directly, while ordinary PyTorch uses __call__.
    output = actor.forward(ids, None, batch(encode_payload()))
    torch.testing.assert_close(compressor.observed[0][0], torch.cat([states[0][1:], states[2][1:]], -1))
    assert output.hidden_states.shape == (4, 4) and output.remory_packed_hidden_states
    assert hook.payload is None and not hook.captured
    summary = actor(ids, None, batch(encode_payload(operation="summary")))
    torch.testing.assert_close(summary.hidden_states, actor.model.norm(states[-1])[1:])
    payload = {"mode": "recover", "positions": [1, 2], "memory": [[7.]*4, [8.]*4]}
    observed = []
    actor.model.layers[0].register_forward_pre_hook(lambda m, a: observed.append(a[0].clone()))
    actor(ids, None, batch(payload))
    torch.testing.assert_close(observed[-1][1:3], torch.tensor([[7.]*4, [8.]*4]))
    actor(ids, None, batch(payload, decode=True))
    torch.testing.assert_close(observed[-1], actor.model.embed_tokens(ids))
    with pytest.raises(RuntimeError, match="prefix reuse"):
        actor(ids, None, batch(encode_payload(), prefix=1))
    assert hook.payload is None
