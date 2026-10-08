import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from remory.client import Client
from remory.server import create_app


def test_compact_resume_generate_delete_api(runtime):
    def send(request):
        response = client.request(request.method, str(request.url),
                                  headers=dict(request.headers), content=request.content)
        return httpx.Response(response.status_code, headers=dict(response.headers), content=response.content)

    headers = {"X-Remory-Session": "a"}
    with TestClient(create_app(runtime)) as client, Client(
        "http://testserver", session_id="a", transport=httpx.MockTransport(send)
    ) as sdk:
        body = {"prefix_ids": [1], "history_ids": [2, 3], "summary_ids": [4]}
        assert client.post("/v1/compact", json=body).status_code == 400
        handle = sdk.compact(**body)["handle"]
        generated = sdk.generate(handle=handle, continuation_ids=[8], max_new_tokens=10)
        assert generated["text"] == "ok"
        assert runtime.backend.calls[-1][0].memory.shape == (4, 3)
        wrong_owner = {**headers, "X-Remory-Session": "b"}
        generate = {"input_ids": [], "remory": {"handle": handle}}
        assert client.post("/generate", headers=wrong_owner, json=generate).status_code == 404
        assert client.post("/v1/compact", headers=headers, json={**body, "image_data": ["x"]}).status_code == 422
        assert client.post("/v1/compact", headers=headers, json={**body, "prefix_ids": [True]}).status_code == 422
        assert sdk.delete(handle) == {"deleted": True}
        assert client.post("/generate", headers=headers, json=generate).status_code == 404


def test_api_backend_failure_does_not_return_checkpoint(runtime):
    runtime.backend.fail = "leaves"
    with TestClient(create_app(runtime)) as client:
        result = client.post("/v1/compact", headers={"X-Remory-Session": "a"},
            json={"prefix_ids": [1], "history_ids": [2], "summary_ids": [3]})
        assert result.status_code == 502
        assert "handle" not in result.json()
