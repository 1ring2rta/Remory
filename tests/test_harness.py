import pytest
from remory.adapters import ChatTemplate, Checkpoint, Harness
from remory.types import digest


class Tokenizer:
    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt, **kwargs):
        assert kwargs["return_dict"] is False
        return [len(m["content"]) for m in messages] + ([99] if add_generation_prompt else [])

    def encode(self, text, **kwargs):
        return [len(text)]


class Client:
    def compact(self, **kwargs):
        self.request = kwargs
        return {"handle": "rm_" + "a" * 48}

    def generate(self, **kwargs):
        return kwargs


def test_harness_commits_and_preserves_native_continuation():
    client = Client()
    harness = Harness(client, ChatTemplate(Tokenizer()))
    prefix = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
    old = [{"role": "assistant", "content": "response"}]
    checkpoint = harness.on_compact(prefix_messages=prefix, removed_messages=old, summary="summary")
    assert client.request["history_ids"] == [8]
    assert Checkpoint.from_dict(checkpoint.to_dict()) == checkpoint
    assert checkpoint.prefix_sha256 == digest([6, 4])
    result = harness.generate(checkpoint=checkpoint, prefix_messages=prefix,
                              messages=[{"role": "user", "content": "next"}])
    assert result["continuation_ids"] == [4, 99]
    with pytest.raises(ValueError, match="prefix changed"):
        harness.generate(checkpoint=checkpoint, prefix_messages=[{"role": "user", "content": "changed"}], messages=[])


def test_chat_template_does_not_silently_drop_images():
    with pytest.raises(ValueError, match="images are unsupported"):
        ChatTemplate(Tokenizer()).split([{"role": "user", "content": [{"type": "image"}]}], [])


def test_compact_generates_summary_before_residual_and_reuses_previous(monkeypatch):
    client = Client()
    harness = Harness(client, ChatTemplate(Tokenizer()))
    prefix = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
    history = [{"role": "assistant", "content": "response"}]
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return {"text": "summary", "meta_info": {"finish_reason": {"type": "stop"}}}
    monkeypatch.setattr(client, "generate", generate)
    first = harness.compact(prefix_messages=prefix, removed_messages=history, summary_prompt="summarize")
    assert calls[0]["input_ids"] == [6, 4, 8, 9, 99]
    assert client.request["history_ids"] == [8]  # Summary instruction is not part of the source.
    assert client.request["summary_ids"] == [7]
    harness.compact(prefix_messages=prefix, removed_messages=history,
                    summary_prompt="summarize", previous=first)
    assert calls[1]["handle"] == first.handle
    assert calls[1]["continuation_ids"] == [8, 9, 99]
    assert client.request["previous"] == first.handle
    previous_request = client.request
    monkeypatch.setattr(client, "generate", lambda **kw: {
        "text": "unfinished", "meta_info": {"finish_reason": {"type": "length"}}})
    with pytest.raises(RuntimeError, match="summary did not complete"):
        harness.compact(prefix_messages=prefix, removed_messages=history,
                        summary_prompt="summarize", previous=first)
    assert client.request is previous_request
