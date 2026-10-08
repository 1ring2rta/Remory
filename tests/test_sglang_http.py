from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
from remory.backends.sglang import SGLangBackend  # noqa: E402
from remory.backends.sglang_http import SGLangHTTPBackend  # noqa: E402
from remory.sglang_server import restore_generation  # noqa: E402
from remory.types import Contract, Prepared  # noqa: E402


def test_residual_encoding_posts_native_sglang_request(runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(Contract, "from_config", lambda *a, **kw: runtime.contract)
    def respond(request):
        import json
        body = json.loads(request.content)
        assert request.url.path == "/generate"
        assert body["input_ids"] == [1, 2, 3]
        assert body["sampling_params"]["max_new_tokens"] == 0
        assert body["sampling_params"]["custom_params"]["remory"]["operation"] == "summary"
        assert body["return_hidden_states"] is True
        return httpx.Response(200, json={"meta_info": {"hidden_states": [[[1., 2., 3.]]]}})
    backend = SGLangHTTPBackend("http://sglang", {"target_revision": "test"}, tmp_path,
        context_limit=4096, transport=httpx.MockTransport(respond))
    try:
        assert backend.encode(Prepared((1, 2, 3)), start=1, end=2, operation="summary").shape == (1, 3)
    finally:
        backend.close()


def test_native_generation_restores_memory_and_checks_session_and_budget(runtime):
    memory = runtime.compact(owner="a", prefix_ids=[1], history_ids=[2, 3], summary_ids=[4])
    backend = SGLangBackend.__new__(SGLangBackend)
    backend.contract, backend.context_limit = runtime.contract, 4096
    runtime.backend = backend
    def request(**overrides):
        return SimpleNamespace(**({"remory": {"handle": memory.handle}, "input_ids": [8],
            "sampling_params": {"max_new_tokens": 4, "top_p": .9},
            "custom_logit_processor": None, **overrides}))
    obj = request()
    restore_generation(runtime, obj, "a")
    expected = runtime._prepare(owner="a", handle=memory.handle, continuation_ids=[8])
    assert obj.input_ids == list(expected.input_ids)
    payload = obj.sampling_params["custom_params"]["remory"]
    assert payload["positions"] == list(expected.memory_positions)
    assert payload["memory"] == memory.embeddings.tolist()
    assert obj.sampling_params["top_p"] == .9
    assert obj.remory is None
    with pytest.raises(KeyError):
        restore_generation(runtime, request(), "b")
    with pytest.raises(ValueError, match="context limit"):
        restore_generation(runtime, request(sampling_params={"max_new_tokens": 4096}), "a")
    with pytest.raises(ValueError, match="cannot be combined"):
        restore_generation(runtime, request(text="ambiguous prompt"), "a")
