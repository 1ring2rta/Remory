import json
import httpx
import numpy as np
import pytest

from remory import Prepared
from remory.backends.sglang import ABI, CAPABILITIES, HOOK, PARAM, SGLangBackend


def config():
    return {"compressor": {"target_hidden_size": 3, "max_depth": 4},
            "residual": {"block_tokens": 8, "compression_ratio": 2, "memory_budget": 8},
            "summary_contract": {"end_token_id": 2, "student_summary_before_ids": [10],
                "student_summary_after_ids": [11], "student_memory_before_ids": [12], "student_memory_after_ids": [13]}}


def server_info():
    return {"request_capabilities": list(CAPABILITIES), "max_running_requests": 1,
            "context_length": 4096, "model_path": "/actor", "forward_hooks": [
                {"hook_factory": HOOK, "config": {"checkpoint": "/checkpoint", "residual_serving_abi": ABI}}]}


def test_real_abi_payload_and_cache_receipt():
    seen = []
    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json=server_info())
        payload = json.loads(request.content)
        seen.append(payload)
        assert payload["cache_policy"] == "bypass"
        assert "custom_logit_processor" in payload
        if payload.get("return_hidden_states"):
            return httpx.Response(200, json={"meta_info": {"cache_policy": "bypass", "hidden_states": [[[1, 2, 3]]]}})
        return httpx.Response(200, json={"text": "ok", "output_ids": [2], "meta_info": {"cache_policy": "bypass"}})
    backend = SGLangBackend("http://worker", config(), server_checkpoint="/checkpoint", server_model="/actor",
                            transport=httpx.MockTransport(respond))
    encoded = backend.encode(Prepared((1, 3)), start=1, end=2, operation="summary")
    np.testing.assert_array_equal(encoded, [[1, 2, 3]])
    source = Prepared((1, 2, 8), (1,), encoded)
    assert backend.generate(source, max_new_tokens=3).text == "ok"
    custom = seen[-1]["sampling_params"]["custom_params"][PARAM]
    assert custom["mode"] == "recover" and custom["memory_embeddings"] == [[1, 2, 3]]
    assert custom["memory_positions"] == [1]
    backend.close()


def test_stock_server_and_missing_bypass_ack_are_rejected():
    with pytest.raises(RuntimeError, match="lacks residual"):
        SGLangBackend("http://worker", config(), server_checkpoint="/checkpoint", server_model="/actor",
                      transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    def respond(request):
        return httpx.Response(200, json=server_info() if request.method == "GET" else {"output_ids": [2], "meta_info": {}})
    backend = SGLangBackend("http://worker", config(), server_checkpoint="/checkpoint", server_model="/actor",
                            transport=httpx.MockTransport(respond))
    with pytest.raises(RuntimeError, match="acknowledge"):
        backend.generate(Prepared((1, 2), (1,), np.ones((1, 3))), max_new_tokens=2)
    backend.close()
