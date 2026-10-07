import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from remory.server import create_app


def test_compact_resume_generate_delete_api(runtime):
    headers = {"Authorization": "Bearer key", "X-Remory-Session": "a"}
    with TestClient(create_app(runtime, api_key="key")) as client:
        assert client.post("/v1/compact", json={}).status_code == 401
        body = {"prefix_ids": [1], "history_ids": [2, 3], "summary_ids": [4]}
        response = client.post("/v1/compact", headers=headers, json=body)
        assert response.status_code == 200, response.text
        handle = response.json()["handle"]
        generated = client.post("/v1/generate", headers=headers,
                                json={"handle": handle, "continuation_ids": [8], "max_new_tokens": 10})
        assert generated.json()["text"] == "ok"
        assert runtime.backend.calls[-1][0].memory.shape == (4, 3)
        wrong_owner = {**headers, "X-Remory-Session": "b"}
        assert client.post("/v1/generate", headers=wrong_owner, json={"handle": handle}).status_code == 404
        assert client.post("/v1/compact", headers=headers, json={**body, "image_data": ["x"]}).status_code == 422
        assert client.post("/v1/compact", headers=headers, json={**body, "prefix_ids": [True]}).status_code == 422
        assert client.delete("/v1/memories/" + handle, headers=headers).status_code == 200
        assert client.post("/v1/generate", headers=headers, json={"handle": handle}).status_code == 404


def test_api_backend_failure_does_not_return_checkpoint(runtime):
    runtime.backend.fail = "leaves"
    with TestClient(create_app(runtime, api_key="key")) as client:
        result = client.post("/v1/compact", headers={"Authorization": "Bearer key", "X-Remory-Session": "a"},
            json={"prefix_ids": [1], "history_ids": [2], "summary_ids": [3]})
        assert result.status_code == 502
        assert "handle" not in result.json()
