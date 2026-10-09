"""Strict safetensors loading of the released inference-only checkpoint."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .compressor import SummaryResidualCompressor
from .layers import build_qwen36_27b_dflash_config

WEIGHTS = "mocoV3/Qwen3.8-27B-REMORY-1.9B"
REVISION = "42f1e7011a75989909d1a28673e6f206f1dee791"


def resolve_checkpoint(checkpoint: str = WEIGHTS, *, revision=REVISION):
    path = Path(checkpoint)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(checkpoint, revision=revision,
                                     allow_patterns=["config.json", "manifest.json", "model.safetensors"]))
    config = json.loads((path / "config.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if hashlib.sha256(canonical.encode()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("checkpoint configuration checksum mismatch")
    return path.resolve(), config


def resolve_actor(config, actor_path=None):
    if actor_path is not None:
        path = Path(actor_path).expanduser().resolve()
        if (path / "config.json").is_file():
            return path
        if str(actor_path) != config["target_model"]:
            raise ValueError("model path must be a local actor snapshot or the checkpoint's target model ID")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(config["target_model"], revision=config["target_revision"],
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"]))


def load_compressor(checkpoint: str = WEIGHTS, *, revision=REVISION, device="cpu", dtype=None):
    path, config = resolve_checkpoint(checkpoint, revision=revision)
    manifest = json.loads((path / "manifest.json").read_text())
    with (path / "model.safetensors").open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != manifest["safetensors_sha256"]:
        raise ValueError("checkpoint weights checksum mismatch")
    c = config["compressor"]
    expected = {"target_hidden_size": 5120, "hidden_size": 5120, "compression_ratio": 16,
                "num_hidden_layers": 5, "target_layer_ids": [1, 16, 31, 46, 61],
                "intermediate_size": 17408, "num_attention_heads": 32,
                "num_key_value_heads": 8, "head_dim": 128,
                "layer_types": ["sliding_attention"] * 4 + ["full_attention"]}
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported checkpoint architecture")
    with torch.device("meta"):
        model = SummaryResidualCompressor(
            build_qwen36_27b_dflash_config(), target_hidden_size=5120,
            target_layer_ids=c["target_layer_ids"], compression_ratio=16,
            local_window=c["local_window"], output_embedding_rms=c["output_embedding_rms"],
            gradient_checkpointing=False, max_depth=c["max_depth"],
            summary_gate_init=c["summary_gate_init"])
    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True, assign=True)
    with torch.device("cpu"):
        model.rotary_emb = type(model.rotary_emb)(model.config, device="cpu")
    model = model.to(device=device, dtype=dtype or torch.float32).requires_grad_(False).eval()
    return model, config
