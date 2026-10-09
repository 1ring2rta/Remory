"""GLM feature capture and soft-memory injection for the pinned SGLang runtime.

Isolated inference adapter; no legacy Qwen residual modules/ABI are reused.
Requires eager, non-overlap TP serving. Source capture is explicitly the
learned attn_hc collapsed INPUT, before the block's attention RMSNorm.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from collections import OrderedDict

import torch
from torch.nn import functional as F

ABI = 'glm53_stage2_memory_v1'
FEATURE_VIEW = 'decoder_block_attn_hc_collapsed_input'
UUID_RE = re.compile(r'^[a-f0-9]{32}$')
_LIVE_ADAPTER = None
_SEEN_REQUEST_IDS = set()
ADAPTER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def validated_id(value):
    if not isinstance(value, str) or not UUID_RE.fullmatch(value):
        raise ValueError('memory/request ID must be a 32-character lowercase hex UUID')
    return value


def validate_http_request(obj):
    """Reject memory cache collisions before request admission or GPU work."""
    custom = obj.sampling_params.get('custom_params') if isinstance(obj.sampling_params, dict) else None
    payload = custom.get('glm53_memory') if isinstance(custom, dict) else None
    if payload is None:
        return
    if not os.environ.get('GLM53_COMPRESSOR_CHECKPOINT'):
        raise ValueError('GLM memory adapter is not configured')
    if payload.get('abi') != ABI or payload.get('mode') not in {'encode', 'recover'}:
        raise ValueError('unsupported GLM memory ABI/mode')
    if any(str(name).startswith('_glm53_http_') for name in payload):
        raise ValueError('reserved HTTP-validated metadata cannot be supplied by callers')
    key = validated_id(payload.get('request_id'))
    if obj.rid != key or obj.cache_salt != 'glm53-' + payload['mode'] + '-' + key:
        raise ValueError('memory requests require matching unique rid and cache_salt')
    ids = obj.input_ids
    if not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids) or payload.get('input_tokens') != len(ids):
        raise ValueError('memory requests require one exact input_ids sequence')
    if getattr(obj, 'input_embeds', None):
        raise ValueError('external input_embeds cannot be mixed with memory adapter overrides')
    if getattr(obj, 'image_data', None):
        positions = [n for desc in payload.get('memories', []) for n in desc['positions']]
        markers = [i for i, token in enumerate(ids) if token in {154854, 154855, 154830, 154832}]
        if payload['mode'] != 'recover' or not positions or not markers or max(positions) >= min(markers):
            raise ValueError('memory plus vision requires every soft slot before the first native image/video boundary')
        # TokenizerManager replaces input_ids with the processor-expanded IDs
        # before creating Req. Req.origin_input_ids_unpadded therefore already
        # contains expanded image tokens: it is not the original HTTP prompt.
        # Preserve only the prefix that must stay fixed for the soft scatter.
        payload['_glm53_http_original_input_tokens'] = len(ids)
        payload['_glm53_http_memory_prefix_ids'] = list(ids[:max(positions) + 1])
    if key in _SEEN_REQUEST_IDS:
        raise ValueError('memory request UUID was already admitted; retries require a fresh UUID')
    root = Path(os.environ['GLM53_MEMORY_CACHE_DIR'])
    if (root / ('consume-' + key + '.json')).exists():
        raise ValueError('memory request UUID already has a committed consumption receipt')
    if payload['mode'] == 'encode':
        memory_id = validated_id(payload.get('memory_id'))
        if (root / (memory_id + '.pt')).exists():
            raise ValueError('memory UUID is already committed')
    _SEEN_REQUEST_IDS.add(key)


def collapsed_attention_input(layer, hidden):
    """Matches Glm5NextTextHyperConnection.forward's `collapsed` exactly."""
    hc, width = int(layer.config.hc_mult), int(layer.config.hidden_size)
    # Native SGLang expands the first block's embedding stream inside its
    # communicator, after the decoder prehook. HF expands before the block.
    if hidden.shape[-1] == width:
        hidden = hidden.repeat(1, hc)
    if hidden.ndim != 2 or hidden.shape[-1] != hc * width:
        raise ValueError('GLM source is neither the initial embedding nor widened mHC stream')
    stream = hidden.reshape(-1, hc, width)
    flat = stream.flatten(1).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + layer.config.rms_norm_eps)
    pre_w = F.linear(flat, layer.hc_attn_fn.float())[:, :hc]
    pre = torch.sigmoid(pre_w * layer.hc_attn_scale[0] + layer.hc_attn_base[:hc]) + layer.config.hc_eps
    return (pre.unsqueeze(-1) * stream).sum(1).to(hidden.dtype)


class Glm53MemoryAdapter:
    def __init__(self, model):
        global _LIVE_ADAPTER
        self.model = model
        self.root = Path(os.environ['GLM53_MEMORY_CACHE_DIR']).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        checkpoint = Path(os.environ['GLM53_COMPRESSOR_CHECKPOINT']).resolve()
        self.config = json.loads((checkpoint / 'config.json').read_text())
        manifest = json.loads((checkpoint / 'manifest.json').read_text())
        if self.config['schema'] != 'remory_glm53_local_evaluation_v1' or self.config['feature_view'] != FEATURE_VIEW:
            raise ValueError('unsupported GLM evaluation checkpoint')
        self.checkpoint_sha256 = manifest['safetensors_sha256']
        from ..models.load import load_compressor
        cc = self.config['compressor']
        self.layers = tuple(cc['target_layer_ids'])
        if self.layers != (0, 11, 22, 33, 44) or cc['target_hidden_size'] != 4096:
            raise ValueError('unexpected GLM Compressor layer geometry')
        self.device = model.model.embed_tokens.weight.device
        self.compressor, _ = load_compressor(checkpoint, device=self.device, dtype=torch.bfloat16)
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.active, self.chunk_features = [], {}
        self.encodes, self.recovers = {}, {}
        self.memory_cache = OrderedDict()
        _LIVE_ADAPTER = self
        self.hooks = [model.model.embed_tokens.register_forward_hook(self._embed_hook)]
        for index in self.layers:
            layer = model.model.layers[index]
            def capture(module, args, *, index=index):
                if not any(x['payload']['mode'] == 'encode' for x in self.active):
                    return
                # Decoder positional args: positions, hidden_states, forward_batch, residual.
                if len(args) < 3:
                    raise RuntimeError('unexpected GLM decoder call ABI')
                hidden = args[1]
                collapsed = collapsed_attention_input(module, hidden)
                expected = sum(x['extend_len'] for x in self.active_all)
                if collapsed.shape != (expected, 4096):
                    raise RuntimeError('capture shape changed: TP token sharding/TBO unsupported by strict adapter')
                self.chunk_features[index] = collapsed
            self.hooks.append(layer.register_forward_pre_hook(capture))
        if self.rank == 0:
            atomic_json(self.root / f'server-{os.environ.get("GLM53_SERVER_ID", "unknown")}.json',
                {'abi': ABI, 'pid': os.getpid(), 'feature_view': FEATURE_VIEW,
                 'target_layer_ids': list(self.layers), 'compressor_sha256': self.checkpoint_sha256,
                 'source_global_step': self.config['source_global_step'], 'ready': True,
                 'adapter_source_sha256': ADAPTER_SOURCE_SHA256,
                 'created_at_unix': time.time(),
                 'capabilities': ['exact_attn_hc_input_capture', 'chunked_source_encode',
                                  'binary_soft_memory_scatter', 'consumption_receipts'],
                 'request_capabilities': ['glm53_stage2_memory_request_isolation_v1']})

    def begin(self, batch):
        self.active, self.active_all, self.chunk_features = [], [], {}
        if not batch.forward_mode.is_extend_without_speculative():
            return
        params = getattr(batch.sampling_info, 'custom_params', None)
        if not params or not any(isinstance(p, dict) and 'glm53_memory' in p for p in params):
            return
        lengths, prefixes = batch.extend_seq_lens_cpu, batch.extend_prefix_lens_cpu
        if not lengths or len(lengths) != len(params):
            raise RuntimeError('request-aligned custom_params/prefill shape mismatch')
        offset = 0
        for row, (length, prefix, custom) in enumerate(zip(lengths, prefixes, params)):
            item = {'row': row, 'offset': offset, 'extend_len': int(length), 'prefix': int(prefix)}
            self.active_all.append(item)
            offset += int(length)
            payload = custom.get('glm53_memory') if isinstance(custom, dict) else None
            if payload is None:
                continue
            if payload.get('abi') != ABI or payload.get('mode') not in {'encode', 'recover'}:
                raise ValueError('unknown GLM memory ABI/mode')
            validated_id(payload.get('request_id'))
            req = custom.get('__req__')
            if req is not None and getattr(req, 'cache_salt', None) != 'glm53-' + payload['mode'] + '-' + payload['request_id']:
                raise ValueError('memory request cache namespace does not match unique request identity')
            if req is not None:
                actual_ids = getattr(req, 'origin_input_ids', None)
                http_prefix = payload.get('_glm53_http_memory_prefix_ids')
                if actual_ids is not None and (len(actual_ids) != payload.get('input_tokens') or http_prefix is not None):
                    if (payload['mode'] != 'recover' or not isinstance(http_prefix, list)
                            or payload.get('_glm53_http_original_input_tokens') != payload.get('input_tokens')
                            or len(actual_ids) < payload['input_tokens']):
                        raise ValueError('memory input length differs without a validated vision expansion')
                    positions = [n for desc in payload.get('memories', []) for n in desc['positions']]
                    last = max(positions, default=-1)
                    if len(http_prefix) != last + 1 or list(actual_ids[:last+1]) != http_prefix:
                        raise ValueError('vision expansion moved or changed soft memory prefix positions')
                    payload = {**payload, 'text_input_tokens': payload['input_tokens'], 'input_tokens': len(actual_ids)}
            input_tokens = payload.get('input_tokens')
            if type(input_tokens) is not int or input_tokens <= 0 or int(prefix) + int(length) > input_tokens:
                raise ValueError('GLM memory prompt extent mismatch')
            if payload['mode'] == 'encode':
                validated_id(payload.get('memory_id'))
                span = payload.get('source_span')
                if not isinstance(span, list) or len(span) != 2 or any(type(n) is not int for n in span) or not 0 < span[0] < span[1] == input_tokens:
                    raise ValueError('encode requires exact post-prefix source span ending at input end')
            memories = payload.get('memories', [])
            used = set()
            for memory in memories:
                validated_id(memory.get('memory_id'))
                positions = memory.get('positions')
                if not isinstance(positions, list) or any(type(n) is not int or n < 0 or n >= input_tokens for n in positions) or positions != sorted(set(positions)):
                    raise ValueError('invalid absolute memory positions')
                if used.intersection(positions):
                    raise ValueError('overlapping memory intervals')
                memory_offset, memory_length = memory.get('offset', 0), memory.get('length', len(positions))
                if type(memory_offset) is not int or memory_offset < 0 or type(memory_length) is not int or memory_length <= 0 or memory_length != len(positions):
                    raise ValueError('invalid memory slice offset/length')
                used.update(positions)
            if payload['mode'] == 'recover' and not used:
                raise ValueError('recover requires nonzero memory slots')
            self.active.append({**item, 'payload': payload})

    def _memory(self, desc):
        key = validated_id(desc['memory_id'])
        sha = desc.get('sha256')
        if not isinstance(sha, str) or not re.fullmatch('[a-f0-9]{64}', sha):
            raise ValueError('memory checksum is required')
        if key in self.memory_cache:
            saved_sha, tensor = self.memory_cache.pop(key)
            if sha != saved_sha:
                raise ValueError('same memory ID used with different checksum')
            self.memory_cache[key] = (sha, tensor)
            return tensor
        path = self.root / (key + '.pt')
        if path.is_symlink() or file_sha256(path) != sha:
            raise ValueError('memory file/checksum mismatch')
        obj = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        tensor = obj['embeddings']
        if obj.get('abi') != ABI or obj.get('memory_id') != key or obj.get('compressor_sha256') != self.checkpoint_sha256 or tensor.ndim != 2 or tensor.shape[1] != 4096 or tensor.dtype != torch.bfloat16:
            raise ValueError('memory binary geometry/provenance mismatch')
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('nonfinite soft memory')
        # Cache CPU storage only; each prefill copies only its intersecting rows.
        self.memory_cache[key] = (sha, tensor)
        while len(self.memory_cache) > 32:
            self.memory_cache.popitem(last=False)
        return tensor

    def _embed_hook(self, module, args, output):
        if not self.active:
            return output
        for item in self.active:
            p = item['payload']
            record = self.recovers.setdefault(p['request_id'], {'positions': set(), 'payload': p})
            for desc in p.get('memories', []):
                tensor = self._memory(desc)
                start, length = desc.get('offset', 0), desc.get('length', len(desc['positions']))
                if start + length > tensor.shape[0]:
                    raise ValueError('memory slice exceeds binary slots')
                tensor = tensor[start:start + length]
                pairs = [(i, pos) for i, pos in enumerate(desc['positions']) if item['prefix'] <= pos < item['prefix'] + item['extend_len']]
                if pairs:
                    rows, positions = zip(*pairs)
                    target = torch.tensor([item['offset'] + pos - item['prefix'] for pos in positions], device=output.device)
                    output[target] = tensor[list(rows)].to(output.device, output.dtype)
                    record['positions'].update(positions)
        return output

    def _compress(self, features, length):
        inputs = features.unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)
        mask = torch.ones((1, length), device=self.device, dtype=torch.bool)
        out = self.compressor(inputs, mask,
            source_token_lengths=torch.tensor([length], device=self.device),
            slot_token_lengths=torch.tensor([1024], device=self.device))
        if out.embeddings.shape != (1, 64, 4096) or not bool(out.attention_mask.all()):
            raise RuntimeError('Compressor slot geometry differs from Stage-2 training')
        return out.embeddings[0].detach().cpu()

    def finish(self):
        for item in self.active:
            p = item['payload']
            final = item['prefix'] + item['extend_len'] == p['input_tokens']
            if p['mode'] == 'encode':
                if set(self.chunk_features) != set(self.layers):
                    raise RuntimeError('missing exact GLM source capture layers')
                start, stop = p['source_span']
                left, right = max(start, item['prefix']), min(stop, item['prefix'] + item['extend_len'])
                state = self.encodes.setdefault(p['request_id'], {'next': start, 'pending': None, 'outputs': [], 'created': time.time()})
                if right > left:
                    if left != state['next']:
                        raise RuntimeError('encode source gap/replay; unique cache salt and full causal capture required')
                    a, b = item['offset'] + left - item['prefix'], item['offset'] + right - item['prefix']
                    features = torch.cat([self.chunk_features[k][a:b] for k in self.layers], dim=-1).detach().cpu()
                    state['next'] = right
                    if state['pending'] is not None:
                        features = torch.cat((state['pending'], features))
                    whole = features.shape[0] // 1024
                    for block in range(whole):
                        state['outputs'].append(self._compress(features[block * 1024:(block + 1) * 1024], 1024))
                    tail = features[whole * 1024:]
                    state['pending'] = tail.clone() if tail.shape[0] else None
                if final:
                    if state['next'] != stop:
                        raise RuntimeError('encode never observed the complete source')
                    if state['pending'] is not None:
                        state['outputs'].append(self._compress(state['pending'], state['pending'].shape[0]))
                    memory = torch.cat(state['outputs'])
                    if self.rank == 0:
                        key = p['memory_id']
                        path = self.root / (key + '.pt')
                        if path.exists():
                            raise FileExistsError('memory UUID already committed')
                        temporary = self.root / ('.' + key + '.pt.tmp')
                        torch.save({'abi': ABI, 'memory_id': key, 'embeddings': memory,
                                    'compressor_sha256': self.checkpoint_sha256}, temporary)
                        os.replace(temporary, path)
                        atomic_json(self.root / (key + '.json'), {
                            'abi': ABI, 'mode': 'encode', 'request_id': p['request_id'],
                            'memory_id': key, 'path': str(path), 'sha256': file_sha256(path),
                            'slots': memory.shape[0], 'hidden_size': 4096, 'dtype': 'bfloat16',
                            'source_span': p['source_span'], 'input_tokens': p['input_tokens'],
                            'source_input_ids_sha256': p.get('source_input_ids_sha256'),
                            'target_layer_ids': list(self.layers), 'feature_view': FEATURE_VIEW,
                            'compressor_sha256': self.checkpoint_sha256, 'source_global_step': self.config['source_global_step'],
                            'memory_built': True, 'complete': True, 'seconds': time.time() - state['created']})
                    del self.encodes[p['request_id']]
            if final:
                record = self.recovers.pop(p['request_id'], {'positions': set()})
                expected = {n for desc in p.get('memories', []) for n in desc['positions']}
                if record['positions'] != expected:
                    raise RuntimeError('soft memory was not injected at every declared position')
                if self.rank == 0 and expected:
                    atomic_json(self.root / ('consume-' + p['request_id'] + '.json'),
                        {'abi': ABI, 'request_id': p['request_id'], 'memory_consumed': True,
                         'memory_slots': len(expected), 'physical_input_tokens': p['input_tokens'],
                         'memories': p['memories'], 'compressor_sha256': self.checkpoint_sha256,
                         'evidence': 'embedding scatter completed and full causal prefill returned',
                         'complete': True})
        self.active, self.chunk_features = [], {}


def install_adapter(cls):
    if not os.environ.get('GLM53_COMPRESSOR_CHECKPOINT'):
        return
    original = cls.forward
    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        adapter = getattr(self, '_glm53_memory_adapter', None)
        if adapter is None:
            adapter = Glm53MemoryAdapter(self)
            self._glm53_memory_adapter = adapter
        adapter.begin(forward_batch)
        try:
            result = original(self, input_ids, positions, forward_batch, *args, **kwargs)
            adapter.finish()
            return result
        finally:
            adapter.active, adapter.chunk_features = [], {}
    cls.forward = forward


def make_attestation_hook(config):
    expected = {'checkpoint', 'compressor_sha256', 'feature_view', 'target_layer_ids', 'abi'}
    if set(config) != expected or config['abi'] != ABI or config['feature_view'] != FEATURE_VIEW:
        raise ValueError('GLM hook attestation declaration is malformed')
    if str(Path(config['checkpoint']).resolve()) != str(Path(os.environ['GLM53_COMPRESSOR_CHECKPOINT']).resolve()):
        raise ValueError('hook declaration does not match loaded checkpoint')
    def verify(module, args, output):
        adapter = _LIVE_ADAPTER
        if adapter is None or adapter.checkpoint_sha256 != config['compressor_sha256'] or list(adapter.layers) != config['target_layer_ids']:
            raise RuntimeError('declared GLM feature hook is not active with the required weights')
        return output
    return verify


def adapter_server_info():
    if not os.environ.get('GLM53_COMPRESSOR_CHECKPOINT'):
        return None
    path = Path(os.environ['GLM53_MEMORY_CACHE_DIR']) / ('server-' + os.environ.get('GLM53_SERVER_ID', 'unknown') + '.json')
    if not path.is_file():
        return {'abi': ABI, 'ready': False, 'reason': 'worker attestation not created'}
    value = json.loads(path.read_text())
    try:
        os.kill(value['pid'], 0)
    except (ProcessLookupError, PermissionError):
        return {'abi': ABI, 'ready': False, 'reason': 'attested worker is not alive'}
    return value
