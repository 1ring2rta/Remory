"""GLM encoding and continuation through the same native SGLang /generate route."""
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

import numpy as np
import torch

from .sglang_http import SGLangHTTPBackend
from .glm_hook import ABI, file_sha256


class GlmSGLangHTTPBackend(SGLangHTTPBackend):
    generation_defaults = {"skip_special_tokens": False, "no_stop_trim": True}

    def __init__(self, url, config, actor, *, cache_dir, weights_sha256, **kwargs):
        super().__init__(url, config, actor, **kwargs)
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.weights_sha256 = weights_sha256
        self.cache_lock = threading.RLock()

    def _memory(self, source):
        # SGLang workers share an internal binary cache with this HTTP process.
        # Clients send memory handles; they never receive server filesystem paths.
        if source.memory is None:
            if source.memory_positions:
                raise ValueError("memory positions require embeddings")
        else:
            a = np.asarray(source.memory, dtype=np.float32)
            if a.shape != (len(source.memory_positions), self.contract.hidden_size) or not np.isfinite(a).all():
                raise ValueError("invalid memory embeddings")
            if (list(source.memory_positions) != sorted(set(source.memory_positions))
                    or any(not 0 <= i < len(source.input_ids) for i in source.memory_positions)):
                raise ValueError("invalid memory positions")
        return {}

    def _descriptor(self, source):
        self._memory(source)
        if source.memory is None:
            return []
        tensor = torch.as_tensor(np.asarray(source.memory)).to(torch.bfloat16).contiguous()
        h = hashlib.sha256(self.weights_sha256.encode())
        h.update(tensor.view(torch.uint8).numpy().tobytes())
        identity = h.hexdigest()
        key = identity[:32]
        path = self.cache / (key + ".pt")
        with self.cache_lock:
            if not path.exists():
                temporary = path.with_name("." + key + ".tmp")
                torch.save({"abi": ABI, "memory_id": key, "embeddings": tensor,
                    "compressor_sha256": self.weights_sha256, "content_sha256": identity}, temporary)
                os.replace(temporary, path)
            else:
                stored = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
                if stored.get("content_sha256") != identity:
                    raise RuntimeError("memory cache identity mismatch")
        return [{"memory_id": key, "sha256": file_sha256(path),
                 "positions": list(source.memory_positions), "offset": 0, "length": len(tensor)}]

    def build_request(self, source, sampling, payload=None):
        if len(source.input_ids) + sampling["max_new_tokens"] > self.context_limit:
            raise ValueError("request exceeds context limit; refusing truncation")
        sampling = {**sampling, "stop_token_ids": list(dict.fromkeys([
            *(sampling.get("stop_token_ids") or []), 154820, 154827, 154829]))}
        key = uuid.uuid4().hex
        request = {"input_ids": list(source.input_ids), "sampling_params": sampling, "rid": key}
        if payload is not None:
            p = {"abi": ABI, "mode": payload["mode"], "request_id": key,
                 "input_tokens": len(source.input_ids), "memories": self._descriptor(source)}
            if p["mode"] == "encode":
                p.update(memory_id=payload["memory_id"], source_span=payload["source_span"],
                    source_input_ids_sha256=hashlib.sha256(json.dumps(
                        list(source.input_ids), separators=(",", ":")).encode()).hexdigest())
            request["sampling_params"] = {**sampling, "custom_params": {"glm53_memory": p}}
            request["cache_salt"] = "glm53-" + p["mode"] + "-" + key
        return request

    def _send(self, body):
        response = self.http.post("/generate", json=body)
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise RuntimeError(f"SGLang request failed: {result['error']}")
        reason = result.get("meta_info", {}).get("finish_reason", {})
        if isinstance(reason, dict) and reason.get("type") in {"error", "abort"}:
            raise RuntimeError("GLM generation was interrupted")
        p = body["sampling_params"].get("custom_params", {}).get("glm53_memory")
        if p and p["memories"]:
            receipt = json.loads((self.cache / ("consume-" + p["request_id"] + ".json")).read_text())
            if (receipt.get("request_id") != p["request_id"] or not receipt.get("memory_consumed")
                    or receipt.get("compressor_sha256") != self.weights_sha256):
                raise RuntimeError("GLM did not confirm memory consumption")
        return result

    def _request(self, source, sampling, payload=None):
        return self._send(self.build_request(source, sampling, payload))

    def encode(self, source, *, start, end, operation, summary=None, input_depth=0, block_depths=()):
        if operation not in {"leaves", "parent"} or not 0 <= start < end == len(source.input_ids):
            raise ValueError("invalid GLM memory source span")
        if summary is not None:
            raise ValueError("the 1.24B GLM encoder does not read summary features")
        if operation == "parent" and (end - start != self.contract.block_tokens
                or source.memory_positions != tuple(range(start, end))):
            raise ValueError("GLM parent encoding requires exactly one block of child memory")
        key = uuid.uuid4().hex
        body = self.build_request(source, {"temperature": 0.0, "max_new_tokens": 1},
            {"mode": "encode", "memory_id": key, "source_span": [start, end]})
        self._send(body)
        receipt = json.loads((self.cache / (key + ".json")).read_text())
        rows = ((end - start + self.contract.block_tokens - 1) // self.contract.block_tokens
                * (self.contract.block_tokens // self.contract.ratio))
        if (receipt.get("request_id") != body["rid"] or receipt.get("slots") != rows
                or receipt.get("compressor_sha256") != self.weights_sha256
                or not receipt.get("complete") or not receipt.get("memory_built")):
            raise RuntimeError("GLM memory encoding receipt is incomplete or mismatched")
        path = self.cache / (key + ".pt")
        if file_sha256(path) != receipt["sha256"]:
            raise RuntimeError("GLM memory encoding checksum mismatch")
        memory = torch.load(path, map_location="cpu", weights_only=True)["embeddings"].float().numpy()
        if memory.shape != (rows, self.contract.hidden_size) or not np.isfinite(memory).all():
            raise RuntimeError("invalid GLM memory encoding")
        # The durable memory is committed to MemoryStore after compaction succeeds.
        path.unlink()
        (self.cache / (key + ".json")).unlink()
        return memory
