"""Exercise the example with both plain-text and structured summary contracts."""
import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest

from remory.client import Client


@pytest.mark.parametrize("structured", [False, True])
def test_quickstart_compacts_with_optional_summary_schema(tmp_path, monkeypatch, structured):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    spec = importlib.util.spec_from_file_location(
        "quickstart_example", Path(__file__).parents[1] / "examples/quickstart.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    contract = {"prompt": "Preserve the next step."}
    schema = {"type": "object", "properties": {"next": {"type": "string"}}}
    if structured:
        contract["output_schema"] = schema
    monkeypatch.setattr(example, "resolve_checkpoint", lambda _: (
        tmp_path, {"summary_contract": contract}))
    monkeypatch.setattr(example, "resolve_actor", lambda *_: tmp_path)

    class Tokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            return [len(m["content"]) for m in messages] + ([99] if add_generation_prompt else [])

        def encode(self, text, **kwargs):
            return [len(text)]

    monkeypatch.setattr(example.AutoTokenizer, "from_pretrained", lambda *_: Tokenizer())
    handle = "rm_" + "a" * 48
    calls = []
    summary = '{"next":"Publish the README."}' if structured else "Publish the README."

    def respond(request):
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path == "/v1/compact":
            assert body["summary_ids"] == [len(summary)]
            return httpx.Response(200, json={"handle": handle})
        if request.method == "DELETE":
            return httpx.Response(200, json={"deleted": True})
        return httpx.Response(200, json={"text": summary,
            "meta_info": {"finish_reason": {"type": "stop"}}})

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(example, "Client", lambda *a, **kw: Client(*a, **kw, transport=transport))
    monkeypatch.setattr(sys, "argv", ["quickstart.py"])
    example.main()
    assert [path for _, path, _ in calls] == [
        "/generate", "/generate", "/v1/compact", "/generate", "/v1/memories/" + handle]
    params = calls[1][2]["sampling_params"]
    if structured:
        assert json.loads(params["json_schema"]) == schema
    else:
        assert "json_schema" not in params
    assert calls[3][2]["remory"] == {"handle": handle}
