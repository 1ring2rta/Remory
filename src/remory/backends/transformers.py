"""Single-request reference backend; no patched serving engine required.

Captures selected actor layer outputs under the complete causal input. Its
unbatched prefill is intended for integration and correctness checks, not the
long-context throughput of the optimized SGLang runtime.
"""
from __future__ import annotations

import threading
import numpy as np
import torch

from ..types import Contract, Generation


class TransformersBackend:
    def __init__(self, actor, compressor, tokenizer, contract: Contract, *, context_limit=32768):
        self.actor = actor.requires_grad_(False).eval()
        self.compressor = compressor.requires_grad_(False).eval()
        self.tokenizer, self.contract, self.context_limit = tokenizer, contract, context_limit
        self.backbone = getattr(actor.model, "language_model", actor.model)
        self.lock = threading.RLock()
        layers = getattr(self.backbone, "layers", ())
        # Last decoder output is normalized in HF's hidden_states convention;
        # published Remory uses interior layers only, so reject that ambiguity.
        if any(i < 0 or i >= len(layers) - 1 for i in compressor.target_layer_ids):
            raise ValueError("reference backend requires interior decoder output layers")
        if actor.get_input_embeddings().weight.shape[1] != contract.hidden_size:
            raise ValueError("actor embedding width differs from compressor")

    @classmethod
    def from_pretrained(cls, checkpoint="mocoV3/Remory-Qwen3.8-27B", *, actor_path=None,
                        device="cuda:0", context_limit=32768):
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        from ..models.load import load_compressor
        compressor, config = load_compressor(checkpoint, device=device, dtype=torch.bfloat16)
        actor_id = actor_path or config["target_model"]
        kwargs = {"revision": config["target_revision"]} if actor_path is None else {}
        actor = AutoModelForImageTextToText.from_pretrained(
            actor_id, dtype=torch.bfloat16, device_map=device, trust_remote_code=False, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(actor_id, trust_remote_code=False, **kwargs)
        contract = Contract.from_config(config, identity=f"transformers:{actor_id}:{config.get('target_revision')}")
        return cls(actor, compressor, tokenizer, contract, context_limit=context_limit)

    def _embed(self, source):
        layer = self.actor.get_input_embeddings()
        ids = torch.tensor([source.input_ids], dtype=torch.long, device=layer.weight.device)
        embeds = layer(ids)
        if source.memory is not None:
            if source.memory.shape != (len(source.memory_positions), self.contract.hidden_size):
                raise ValueError("invalid memory geometry")
            embeds[:, list(source.memory_positions), :] = torch.as_tensor(
                source.memory, device=embeds.device, dtype=embeds.dtype)
        return embeds

    def encode(self, source, *, start, end, operation, summary=None, input_depth=0, block_depths=()):
        if operation not in {"summary", "leaves", "recursive_leaves", "parent"}:
            raise ValueError("unsupported residual operation")
        if not 0 <= start < end <= len(source.input_ids) <= self.context_limit:
            raise ValueError("invalid source span or context length")
        captured, hooks = {}, []
        with self.lock, torch.inference_mode():
            if operation != "summary":
                for index in self.compressor.target_layer_ids:
                    def capture(module, args, output, index=index):
                        value = output[0] if isinstance(output, tuple) else output
                        captured[index] = value[:, start:end].detach().to("cpu")
                    hooks.append(self.backbone.layers[index].register_forward_hook(capture))
            try:
                embeds = self._embed(source)
                output = self.backbone(inputs_embeds=embeds,
                    attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device),
                    use_cache=False, return_dict=True)
            finally:
                for hook in hooks:
                    hook.remove()
            if operation == "summary":
                return output.last_hidden_state[0, start:end].float().cpu().numpy()
            del output, embeds
            if summary is None or not len(summary):
                raise ValueError("residual encoding requires summary states")
            param = next(self.compressor.parameters())
            summary_tensor = torch.as_tensor(summary, device=param.device, dtype=param.dtype)[None]
            summary_mask = torch.ones(summary_tensor.shape[:2], dtype=torch.bool, device=param.device)
            result = []
            for block_index, left in enumerate(range(0, end - start, self.contract.block_tokens)):
                right = min(end - start, left + self.contract.block_tokens)
                features = torch.cat([captured[i][:, left:right] for i in self.compressor.target_layer_ids], -1)
                features = features.to(device=param.device, dtype=param.dtype)
                mask = torch.ones(features.shape[:2], dtype=torch.bool, device=param.device)
                encoded = self.compressor(features, mask,
                    source_token_lengths=torch.tensor([right - left], device=param.device),
                    slot_token_lengths=torch.tensor([self.contract.block_tokens], device=param.device),
                    summary_features=summary_tensor, summary_attention_mask=summary_mask,
                    input_depth=block_depths[block_index] if operation == "recursive_leaves" else input_depth)
                result.append(encoded.embeddings[encoded.attention_mask].float().cpu().numpy())
            return np.concatenate(result)

    def generate(self, source, *, max_new_tokens, sampling=None):
        if sampling and set(sampling) - {"temperature", "top_p", "top_k", "repetition_penalty"}:
            raise ValueError("unsupported Transformers sampling options")
        if len(source.input_ids) + max_new_tokens > self.context_limit:
            raise ValueError("request exceeds context limit")
        params = dict(sampling or {})
        temperature = params.pop("temperature", 0.0)
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        if temperature > 0:
            params.update(temperature=temperature)
        with self.lock, torch.inference_mode():
            embeds = self._embed(source)
            output = self.actor.generate(inputs_embeds=embeds,
                attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device),
                max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                eos_token_id=self.contract.placeholder_id, pad_token_id=self.contract.placeholder_id,
                use_cache=True, **params)
            ids = output[0].tolist()
        return Generation(self.tokenizer.decode(ids, skip_special_tokens=False), ids,
                          {"prompt_tokens": len(source.input_ids), "completion_tokens": len(ids)},
                          "stop" if ids and ids[-1] == self.contract.placeholder_id else "length")
