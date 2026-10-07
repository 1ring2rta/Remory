"""Client for the tested SummaryResidual SGLang ABI (stock SGLang is insufficient)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import uuid

import httpx
import numpy as np

from ..types import Contract, Generation, Prepared, token_ids

ABI = "qwen38_summary_residual_sglang_v1"
HOOK = "context_compression.qwen38_residual_sglang_hook:make_summary_residual_compressor_hook"
PARAM = "specforge_context_compressor"
CAPABILITIES = {ABI, "qwen38_recursive_summary_residual_v1",
                "qwen38_residual_dynamic_summary_v1", "request_cache_bypass_v1"}


def metadata_processor() -> str:
    # A fixed pickle GLOBAL reference, not serialized local bytecode. The server
    # already ships this no-op class. This client never unpickles remote input.
    reference = (b"ccontext_compression.sglang_compressor_hook\n"
                 b"SpecForgeContextCompressorNoOpLogitProcessor\n.")
    return json.dumps({"callable": reference.hex()})


class SGLangBackend:
    def __init__(self, base_url: str, config: dict, *, server_checkpoint: str,
                 server_model: str, api_key: str | None = None, timeout: float = 600,
                 transport: httpx.BaseTransport | None = None):
        self.config = config
        self.server_checkpoint, self.server_model = server_checkpoint, server_model
        self.http = httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=timeout,
                                 headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                                 transport=transport)
        self.contract = Contract.from_config(config, identity=json.dumps(
            {"checkpoint": server_checkpoint, "actor": server_model}, sort_keys=True))
        try:
            self.attest()
        except Exception:
            self.http.close()
            raise

    def close(self):
        self.http.close()

    def attest(self) -> dict:
        response = self.http.get("get_server_info")
        response.raise_for_status()
        info = response.json()
        caps = set(info.get("request_capabilities") or [])
        if not CAPABILITIES <= caps:
            raise RuntimeError(f"SGLang lacks residual capabilities: {sorted(CAPABILITIES - caps)}")
        concurrency = info.get("max_running_requests")
        if type(concurrency) is not int or concurrency < 1 or (
            concurrency > 1 and "qwen38_residual_cpu_concurrent_v1" not in caps
            and "soft_memory_prefix_reuse_batch_v1" not in caps
        ):
            raise RuntimeError("SGLang does not attest safe soft-memory batching")
        hooks = [h.get("config", {}) for h in info.get("forward_hooks", [])
                 if h.get("hook_factory") == HOOK]
        if len(hooks) != 1 or hooks[0].get("residual_serving_abi") != ABI:
            raise RuntimeError("missing or incompatible residual hook")
        # Compare server namespace strings, not client-side Path.resolve(): the
        # client and worker may be on different hosts or in different containers.
        if hooks[0].get("checkpoint") != self.server_checkpoint:
            raise RuntimeError("residual checkpoint differs from configured server checkpoint")
        if info.get("model_path") != self.server_model:
            raise RuntimeError("actor differs from configured server model")
        limits = [info.get(k) for k in ("context_length", "max_total_tokens",
                                      "max_total_num_tokens", "max_req_input_len")]
        for state in info.get("internal_states", []) or []:
            limits.extend(state.get(k) for k in ("max_total_num_tokens", "max_req_input_len"))
        limits = [int(x) for x in limits if type(x) in (int, float) and math.isfinite(x) and x > 1]
        if not limits:
            raise RuntimeError("backend did not advertise its context ceiling")
        self.context_limit = min(limits) - 1
        return {"abi": ABI, "context_limit": self.context_limit,
                "identity": self.contract.identity, "capabilities": sorted(caps)}

    def _request(self, source: Prepared, sampling: dict, *, custom=None, encode=False) -> dict:
        self.attest()  # Recheck after worker restarts; never quietly use a stock server.
        if len(source.input_ids) + sampling["max_new_tokens"] > self.context_limit:
            raise ValueError("request exceeds backend context ceiling; refusing truncation")
        payload = {"input_ids": list(source.input_ids), "sampling_params": dict(sampling),
                   "rid": "remory-" + uuid.uuid4().hex}
        if custom:
            payload["sampling_params"]["custom_params"] = {PARAM: custom}
            payload.update(custom_logit_processor=metadata_processor(), cache_policy="bypass",
                           extra_key="remory-" + uuid.uuid4().hex)
        if encode:
            payload["return_hidden_states"] = True
        try:
            response = self.http.post("generate", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException:
            try:
                self.http.post("abort_request", json={"rid": payload["rid"]}, timeout=5)
            except httpx.HTTPError:
                pass
            raise
        result = response.json()
        meta = result.get("meta_info", {})
        if custom and meta.get("cache_policy") != "bypass":
            raise RuntimeError("server did not acknowledge cache bypass for soft memory")
        reason = meta.get("finish_reason")
        if isinstance(reason, dict) and reason.get("type") in {"abort", "aborted", "cancelled", "error"}:
            raise RuntimeError("backend aborted residual inference")
        return result

    def _memory_fields(self, source):
        if source.memory is None:
            if source.memory_positions:
                raise ValueError("memory positions require embeddings")
            return {}
        a = np.asarray(source.memory, dtype=np.float32)
        p = source.memory_positions
        if (a.shape != (len(p), self.contract.hidden_size) or not p
                or not np.isfinite(a).all() or tuple(sorted(set(p))) != p
                or any(type(i) is not int or not 0 <= i < len(source.input_ids) for i in p)):
            raise ValueError("invalid sparse memory overrides")
        return {"memory_positions": list(p), "memory_embeddings": a.tolist()}

    def encode(self, source, *, start, end, operation, summary=None, input_depth=0, block_depths=()):
        if operation not in {"summary", "leaves", "recursive_leaves", "parent"}:
            raise ValueError("unsupported residual operation")
        if not 0 <= start < end <= len(source.input_ids):
            raise ValueError("invalid source span")
        residual = {"abi": ABI, "operation": operation, "input_depth": input_depth}
        if summary is not None:
            residual["summary_features"] = np.asarray(summary, dtype=np.float32).tolist()
        if block_depths:
            residual["block_input_depths"] = list(block_depths)
        custom = {"version": 1, "mode": "encode", "evidence_spans": [[start, end]],
                  "source_token_lengths": [end - start], "residual": residual,
                  **self._memory_fields(source)}
        result = self._request(source, {"temperature": 0.0, "max_new_tokens": 0},
                               custom=custom, encode=True)
        chunks = result.get("meta_info", {}).get("hidden_states")
        if not isinstance(chunks, list) or len(chunks) != 1:
            raise RuntimeError("backend did not return a single packed hidden-state matrix")
        return np.asarray(chunks[0], dtype=np.float32)

    def generate(self, source, *, max_new_tokens, sampling=None):
        allowed = {"temperature", "top_p", "top_k", "min_p", "presence_penalty",
                   "repetition_penalty", "sampling_seed", "json_schema"}
        if set(sampling or {}) - allowed:
            raise ValueError("unsupported sampling options")
        params = {"temperature": 0.0, **(sampling or {}), "max_new_tokens": max_new_tokens,
                  "stop_token_ids": [self.contract.placeholder_id],
                  "skip_special_tokens": False, "no_stop_trim": True}
        memory = self._memory_fields(source)
        custom = {"version": 1, "mode": "recover", **memory} if memory else None
        result = self._request(source, params, custom=custom)
        output = token_ids(result.get("output_ids"), "output_ids", empty=True)
        meta = result.get("meta_info", {})
        return Generation(result.get("text", ""), list(output),
                          {k: meta[k] for k in ("prompt_tokens", "completion_tokens", "cached_tokens")
                           if k in meta}, meta.get("finish_reason"))


def from_config_file(path: str | Path, **kwargs) -> SGLangBackend:
    return SGLangBackend(config=json.loads(Path(path).read_text()), **kwargs)
