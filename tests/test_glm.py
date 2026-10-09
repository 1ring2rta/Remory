from types import SimpleNamespace

import numpy as np
import pytest

from remory.adapters import GlmChatTemplate
from remory.glm_engine import GlmRemory
from remory.store import MemoryStore
from remory.types import Contract


class Backend:
    contract = Contract(3, 16, 4, 16, 8, 0, (90,), (), (91,), (), "glm-test")
    context_limit = 4096

    def __init__(self):
        self.calls = []

    def encode(self, source, *, start, end, operation, **kwargs):
        assert "summary" not in kwargs
        assert operation in {"leaves", "parent"}
        self.calls.append((source, start, end, operation))
        count = ((end - start + 15) // 16) * 4
        return np.full((count, 3), len(self.calls), dtype=np.float32)


def engine(tmp_path):
    return GlmRemory(Backend(), MemoryStore(tmp_path / "memory.sqlite"))


def test_glm_reuses_old_frontier_and_excludes_summary_from_encoding(tmp_path):
    e = engine(tmp_path)
    old = e.compact(owner="a", prefix_ids=[1], history_ids=[2] * 9, summary_ids=[999])
    new = e.compact(owner="a", prefix_ids=[1], history_ids=[3] * 5,
                    summary_ids=[998], previous=old.handle)
    source, start, end, operation = e.backend.calls[-1]
    assert source.input_ids == (1, 0, 0, 0, 0, 3, 3, 3, 3, 3)
    assert (start, end, operation) == (5, 10, "leaves")
    assert source.memory_positions == (1, 2, 3, 4)
    np.testing.assert_array_equal(source.memory, old.embeddings)
    np.testing.assert_array_equal(new.embeddings[:4], old.embeddings)
    assert new.receipt["source_tokens"] == 14
    assert new.receipt["source_memory_slots"] == 4
    assert new.slot_depths == (1,) * 8
    restored = e._prepare(owner="a", handle=new.handle, continuation_ids=[5])
    assert restored.input_ids[:4] == (1, 90, 998, 91)


def test_glm_parent_merges_preserve_complete_chronological_coverage(tmp_path):
    e = engine(tmp_path)
    old = e.compact(owner="a", prefix_ids=[1], history_ids=[2] * 64, summary_ids=[9])
    new = e.compact(owner="a", prefix_ids=[1], history_ids=[3] * 5,
                    summary_ids=[8], previous=old.handle)
    parent, start, end, operation = e.backend.calls[-1]
    assert operation == "parent" and (start, end) == (1, 17)
    np.testing.assert_array_equal(parent.memory, old.embeddings)
    assert new.receipt["frontier_nodes"] == [
        {"depth": 2, "start": 0, "stop": 64}, {"depth": 1, "start": 64, "stop": 69}]
    assert new.slot_depths == (2,) * 4 + (1,) * 4


def test_glm_summary_only_update_does_not_reencode_and_failure_keeps_old_memory(tmp_path):
    e = engine(tmp_path)
    old = e.compact(owner="a", prefix_ids=[1], history_ids=[2], summary_ids=[3])
    new = e.compact(owner="a", prefix_ids=[1], history_ids=[], summary_ids=[4], previous=old.handle)
    assert len(e.backend.calls) == 1
    np.testing.assert_array_equal(old.embeddings, new.embeddings)
    with pytest.raises(ValueError, match="prefix changed"):
        e.compact(owner="a", prefix_ids=[9], history_ids=[2], summary_ids=[4], previous=old.handle)
    with pytest.raises(KeyError):
        e.compact(owner="other", prefix_ids=[1], history_ids=[2], summary_ids=[4], previous=old.handle)
    assert e.store.get("a", old.handle).summary_ids == (3,)


def test_glm_summary_requires_a_finished_thinking_boundary():
    codec = GlmChatTemplate(None)
    assert codec.final_text("reasoning</think> checkpoint<|user|>") == "checkpoint"
    with pytest.raises(RuntimeError, match="did not finish"):
        codec.final_text("unfinished reasoning")


def test_glm_recovery_uses_native_ids_and_checks_binary_memory(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from remory.backends.glm_http import GlmSGLangHTTPBackend
    from remory.backends.glm_hook import validate_http_request, Glm53MemoryAdapter
    from remory.sglang_server import restore_generation
    e = engine(tmp_path)
    saved = e.compact(owner="a", prefix_ids=[1], history_ids=[2], summary_ids=[3])
    monkeypatch.setattr(Contract, "from_config", lambda *a, **kw: e.contract)
    backend = GlmSGLangHTTPBackend("http://unused", {"target_revision": "test"}, tmp_path,
        context_limit=4096, cache_dir=tmp_path / "cache", weights_sha256="a" * 64)
    try:
        e.backend = backend
        obj = SimpleNamespace(remory={"handle": saved.handle}, input_ids=[8],
            sampling_params={"max_new_tokens": 4}, custom_logit_processor=None)
        restore_generation(e, obj, "a")
        assert obj.custom_logit_processor is None
        assert obj.sampling_params["stop_token_ids"] == [0, 154820, 154827, 154829]
        p = obj.sampling_params["custom_params"]["glm53_memory"]
        assert obj.rid == p["request_id"]
        assert obj.cache_salt == "glm53-recover-" + obj.rid
        desc = p["memories"][0]
        record = torch.load(backend.cache / (desc["memory_id"] + ".pt"), weights_only=True)
        np.testing.assert_array_equal(record["embeddings"].float().numpy(), saved.embeddings)
        monkeypatch.setenv("GLM53_COMPRESSOR_CHECKPOINT", str(tmp_path))
        monkeypatch.setenv("GLM53_MEMORY_CACHE_DIR", str(backend.cache))
        validate_http_request(obj)
        with pytest.raises(ValueError, match="already admitted"):
            validate_http_request(obj)
        adapter = Glm53MemoryAdapter.__new__(Glm53MemoryAdapter)
        from collections import OrderedDict
        adapter.root, adapter.memory_cache = backend.cache, OrderedDict()
        adapter.checkpoint_sha256 = "a" * 64
        # This tiny fixture must fail the actual adapter's 4096-wide ABI.
        with pytest.raises(ValueError):
            adapter._memory(desc)
    finally:
        backend.close()
