import json
import os
import sys
from types import SimpleNamespace

import pytest

from remory import launch_server


def test_launcher_detects_installed_cuda_and_state_directory(tmp_path, monkeypatch):
    cuda = tmp_path / "cuda"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "bin/ptxas").touch()
    (tmp_path / "installation.json").write_text(json.dumps({"cuda_home": str(cuda)}))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})
    assert launch_server.configure_runtime() == tmp_path
    assert os.environ["CUDA_HOME"] == str(cuda)
    assert os.environ["TRITON_PTXAS_PATH"] == str(cuda / "bin/ptxas")
    assert os.environ["FLASHINFER_WORKSPACE_BASE"] == str(tmp_path / "kernel-cache")
    assert os.environ["PATH"] == str(cuda / "bin") + os.pathsep + "/usr/bin"


@pytest.mark.parametrize("flags", [
    ["--model-path", "--remory-checkpoint", "--context-length", "--mem-fraction-static", "--base-gpu-id"],
    ["--actor", "--checkpoint", "--context-limit", "--memory-fraction", "--gpu"],
])
def test_sglang_flags_and_existing_aliases_launch_the_same_server(tmp_path, monkeypatch, flags):
    seen = []
    monkeypatch.setattr(launch_server, "configure_runtime", lambda _: tmp_path)
    monkeypatch.setitem(sys.modules, "remory.sglang_server", SimpleNamespace(launch=seen.append))
    values = ["Qwen/Qwen3.8-27B", "memory/path", "8192", "0.7", "1"]
    launch_server.main([item for pair in zip(flags, values) for item in pair])
    args = seen[0]
    assert (args.actor, args.checkpoint, args.context_limit, args.memory_fraction, args.gpu) == (
        values[0], values[1], 8192, .7, 1)
    assert args.store == str(tmp_path / "memory.sqlite")
    seen.clear()
    launch_server.main([])
    assert seen[0].context_limit is None


def test_explicit_model_id_keeps_checkpoint_revision(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from remory.models.load import resolve_actor
    config = {"target_model": "Qwen/Qwen3.8-27B", "target_revision": "pinned"}
    seen = []
    def download(model, **kwargs):
        seen.append((model, kwargs["revision"]))
        return str(tmp_path)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert resolve_actor(config, config["target_model"]) == tmp_path
    assert seen == [(config["target_model"], "pinned")]
    (tmp_path / "config.json").write_text("{}")
    assert resolve_actor(config, tmp_path) == tmp_path
    assert len(seen) == 1
