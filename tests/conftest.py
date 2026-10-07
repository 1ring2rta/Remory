import numpy as np
import pytest

from remory import Contract, Generation, MemoryStore, Remory


class RecordingBackend:
    def __init__(self):
        self.contract = Contract(3, 8, 2, 8, 4, 2, (10,), (11,), (12,), (13,), "test-weights")
        self.context_limit = 4096
        self.calls = []
        self.fail = None

    def encode(self, source, **kwargs):
        self.calls.append((source, kwargs))
        op = kwargs["operation"]
        if op == self.fail:
            raise RuntimeError("injected encode failure")
        rows = kwargs["end"] - kwargs["start"]
        if op != "summary":
            rows = ((rows + 7) // 8) * 4
        value = len(self.calls) + (float(source.memory.mean()) if source.memory is not None else 0)
        return np.full((rows, 3), value, dtype=np.float32)

    def generate(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return Generation("ok", [2], {"prompt_tokens": len(source.input_ids)})


@pytest.fixture
def runtime(tmp_path):
    backend = RecordingBackend()
    return Remory(backend, MemoryStore(tmp_path / "memory.sqlite"))


def compact(engine, **overrides):
    return engine.compact(**dict(owner="session-a", prefix_ids=[1, 3], history_ids=list(range(20)),
                                  summary_ids=[50, 51], **overrides))
