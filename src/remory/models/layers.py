# Adapted from Recursive Memory (MIT); see THIRD_PARTY_NOTICES.md.

from __future__ import annotations
import copy
import math
from typing import Any, NamedTuple, Optional, Sequence
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding
DEFAULT_QWEN35_TARGET_LAYER_IDS = (1, 5, 9, 13, 17, 21, 25, 29)
DEFAULT_QWEN36_27B_TARGET_LAYER_IDS = (1, 16, 31, 46, 61)

def build_qwen36_27b_dflash_config(
    *,
    target_hidden_size: int = 5120,
    vocab_size: int = 248320,
    compression_ratio: int = 16,
) -> Qwen3Config:
    """Return the public Qwen3.6-27B-DFlash backbone geometry.

    The published draft checkpoint is used only as an architecture contract;
    context-compression experiments initialize every trainable tensor from
    scratch.  ``block_size`` is recorded as the nominal physical compression
    ratio while the number of output slots remains data dependent.
    """

    for name, value in (
        ("target_hidden_size", target_hidden_size),
        ("vocab_size", vocab_size),
        ("compression_ratio", compression_ratio),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000.0,
        },
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=True,
        sliding_window=2048,
        max_window_layers=5,
        layer_types=["sliding_attention"] * 4 + ["full_attention"],
    )
    config.num_target_layers = 64
    config.target_hidden_size = int(target_hidden_size)
    config.block_size = int(compression_ratio)
    config.dflash_config = {
        "block_size": int(compression_ratio),
        "mask_token_id": 248070,
        "target_layer_ids": list(DEFAULT_QWEN36_27B_TARGET_LAYER_IDS),
    }
    return config



def _rotate_half(hidden: torch.Tensor) -> torch.Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)



def _apply_rotary(
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return hidden * cos.unsqueeze(1) + _rotate_half(hidden) * sin.unsqueeze(1)



def _repeat_kv(hidden: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden
    batch, key_heads, length, head_dim = hidden.shape
    hidden = hidden[:, :, None, :, :].expand(
        batch, key_heads, repeats, length, head_dim
    )
    return hidden.reshape(batch, key_heads * repeats, length, head_dim)



class CompressionAttentionLayout(NamedTuple):
    """Masks and compact positions for one padded evidence batch."""

    local_keep_mask: torch.Tensor
    global_keep_mask: torch.Tensor
    slot_attention_mask: torch.Tensor
    slot_counts: torch.Tensor
    source_position_ids: torch.Tensor
    slot_position_ids: torch.Tensor



def build_local_compression_layout(
    source_attention_mask: torch.Tensor,
    *,
    compression_ratio: int = 16,
    local_window: int = 256,
    source_token_lengths: Optional[torch.Tensor] = None,
    slot_token_lengths: Optional[torch.Tensor] = None,
) -> CompressionAttentionLayout:
    """Build overlapping local-read and global-slot masks.

    Source padding may occur anywhere: locality is defined over the compact
    rank of valid evidence tokens.  Slots are left padded in the returned
    batch, and invalid query rows are allowed to read only their own slot key.
    That diagonal escape hatch prevents all-masked SDPA rows; the compressor
    explicitly zeros those rows after every decoder layer.
    """

    if not isinstance(source_attention_mask, torch.Tensor):
        raise TypeError("source_attention_mask must be a torch.Tensor")
    if source_attention_mask.ndim != 2:
        raise ValueError("source_attention_mask must have shape [batch, source]")
    for name, value in (
        ("compression_ratio", compression_ratio),
        ("local_window", local_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if local_window < compression_ratio:
        raise ValueError(
            "local_window must be at least compression_ratio so every source "
            "token is visible to its compression slot"
        )

    source_mask = source_attention_mask.bool()
    device = source_mask.device
    batch, source_width = source_mask.shape
    source_span_lengths = source_mask.sum(dim=-1)
    if source_token_lengths is None:
        source_lengths = source_span_lengths
    else:
        if (
            not isinstance(source_token_lengths, torch.Tensor)
            or source_token_lengths.ndim != 1
            or source_token_lengths.shape[0] != batch
        ):
            raise ValueError("source_token_lengths must have shape [batch]")
        if source_token_lengths.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("source_token_lengths must use an integer dtype")
        if source_token_lengths.device != device:
            raise ValueError("source_token_lengths and source mask must share a device")
        source_lengths = source_token_lengths.long()
        if bool((source_lengths < 0).any().item()):
            raise ValueError("source_token_lengths must be nonnegative")
        if bool(((source_lengths == 0) != (source_span_lengths == 0)).any().item()):
            raise ValueError(
                "source_token_lengths and valid source spans must be empty together"
            )
    if slot_token_lengths is None:
        slot_lengths = source_lengths
    else:
        if (
            not isinstance(slot_token_lengths, torch.Tensor)
            or slot_token_lengths.ndim != 1
            or slot_token_lengths.shape[0] != batch
        ):
            raise ValueError("slot_token_lengths must have shape [batch]")
        if slot_token_lengths.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("slot_token_lengths must use an integer dtype")
        if slot_token_lengths.device != device:
            raise ValueError(
                "slot_token_lengths and source mask must share a device"
            )
        slot_lengths = slot_token_lengths.long()
        if bool((slot_lengths < source_lengths).any().item()):
            raise ValueError(
                "slot_token_lengths cannot be shorter than source_token_lengths"
            )
        if bool(((slot_lengths == 0) != (source_lengths == 0)).any().item()):
            raise ValueError(
                "slot and source token lengths must be empty together"
            )
        if bool((slot_lengths > source_lengths).any().item()) and local_window < 2048:
            raise ValueError(
                "padded slot allocation requires local_window >= 2048"
            )
    slot_counts = torch.div(
        slot_lengths + compression_ratio - 1,
        compression_ratio,
        rounding_mode="floor",
    )
    slot_width = int(slot_counts.max().item()) if batch else 0

    source_ranks = source_mask.long().cumsum(dim=-1) - 1
    source_ranks.masked_fill_(~source_mask, 0)
    # JSON/chat-template escaping can make the encoded source span wider than
    # the canonical raw stdout used to define the compression ratio.  Map each
    # feature-row centre into canonical-token coordinates so locality and RoPE
    # remain tied to the actual evidence rather than serialization overhead.
    span_denominator = source_span_lengths.clamp_min(1).to(torch.float32)[:, None]
    canonical_scale = source_lengths.to(torch.float32)[:, None] / span_denominator
    source_coordinates = (source_ranks.to(torch.float32) + 0.5) * canonical_scale - 0.5
    source_coordinates.masked_fill_(~source_mask, 0.0)
    source_positions = source_coordinates.round().long()
    maximum_position = (source_lengths - 1).clamp_min(0)[:, None]
    source_positions = torch.minimum(source_positions.clamp_min(0), maximum_position)
    source_positions.masked_fill_(~source_mask, 0)
    if slot_width == 0:
        empty_slots = torch.zeros((batch, 0), dtype=torch.bool, device=device)
        return CompressionAttentionLayout(
            local_keep_mask=torch.zeros(
                (batch, 1, 0, source_width), dtype=torch.bool, device=device
            ),
            global_keep_mask=torch.zeros(
                (batch, 1, 0, 0), dtype=torch.bool, device=device
            ),
            slot_attention_mask=empty_slots,
            slot_counts=slot_counts,
            source_position_ids=source_positions,
            slot_position_ids=torch.zeros((batch, 0), dtype=torch.long, device=device),
        )

    padded_slot_index = torch.arange(slot_width, device=device)[None, :]
    left_padding = slot_width - slot_counts[:, None]
    slot_rank = padded_slot_index - left_padding
    slot_mask = slot_rank >= 0

    # Each slot is anchored at the centre of the evidence block it represents.
    # In particular, centre the final partial block over its *actual* extent.
    # Clipping a full-stride centre to the final token would move a 1:1024 slot
    # hundreds of positions away from a short source and can leave it with no
    # source keys at all.
    block_start = slot_rank.clamp_min(0) * compression_ratio
    final_position = (slot_lengths - 1).clamp_min(0)[:, None]
    block_end = torch.minimum(
        block_start + compression_ratio - 1,
        final_position,
    )
    anchor_float_2d = (
        block_start.to(torch.float32) + block_end.to(torch.float32)
    ) / 2.0
    slot_positions = torch.floor(anchor_float_2d + 0.5).long()
    slot_positions.masked_fill_(~slot_mask, 0)

    half_window = local_window / 2.0
    source_coordinate = source_coordinates[:, None, :]
    anchor_float = anchor_float_2d[:, :, None]
    source_local = (
        source_mask[:, None, :]
        & slot_mask[:, :, None]
        & ((source_coordinate - anchor_float).abs() < half_window)
    )

    # Neighbouring slots exchange information within the same raw-token
    # receptive field.  The sixth layer later performs the sole global mix.
    query_anchor = anchor_float
    key_anchor = anchor_float.transpose(1, 2)
    slot_local = (
        slot_mask[:, :, None]
        & slot_mask[:, None, :]
        & ((query_anchor - key_anchor).abs() < half_window)
    )
    local_keep = torch.cat((source_local, slot_local), dim=-1).unsqueeze(1)

    global_keep = (slot_mask[:, :, None] & slot_mask[:, None, :]).unsqueeze(1)

    # Safe self-unmask for invalid padded query rows.
    diagonal = torch.eye(slot_width, dtype=torch.bool, device=device)[None, :, :]
    invalid_diagonal = (~slot_mask)[:, :, None] & diagonal
    local_keep[:, :, :, source_width:] |= invalid_diagonal.unsqueeze(1)
    global_keep |= invalid_diagonal.unsqueeze(1)

    return CompressionAttentionLayout(
        local_keep_mask=local_keep,
        global_keep_mask=global_keep,
        slot_attention_mask=slot_mask,
        slot_counts=slot_counts,
        source_position_ids=source_positions,
        slot_position_ids=slot_positions,
    )



class GrepSeekDFlashAttention(nn.Module):
    """Native-shape DFlash attention with an explicit compression mask."""

    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config.hidden_size)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(
            getattr(config, "head_dim", self.hidden_size // self.num_attention_heads)
        )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        bias = bool(getattr(config, "attention_bias", False))
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        eps = float(config.rms_norm_eps)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        source_hidden: Optional[torch.Tensor],
        keep_mask: torch.Tensor,
        slot_cos: torch.Tensor,
        slot_sin: torch.Tensor,
        source_cos: Optional[torch.Tensor],
        source_sin: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, slots, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, slots, self.num_attention_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        query = _apply_rotary(query, slot_cos, slot_sin)

        slot_key = self.k_proj(hidden_states).view(
            batch, slots, self.num_key_value_heads, self.head_dim
        )
        slot_key = self.k_norm(slot_key).transpose(1, 2)
        slot_key = _apply_rotary(slot_key, slot_cos, slot_sin)
        slot_value = (
            self.v_proj(hidden_states)
            .view(batch, slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        if source_hidden is None:
            key = slot_key
            value = slot_value
        else:
            source_width = source_hidden.shape[1]
            source_key = self.k_proj(source_hidden).view(
                batch, source_width, self.num_key_value_heads, self.head_dim
            )
            source_key = self.k_norm(source_key).transpose(1, 2)
            if source_cos is None or source_sin is None:
                raise ValueError("source RoPE tensors are required with source_hidden")
            source_key = _apply_rotary(source_key, source_cos, source_sin)
            source_value = (
                self.v_proj(source_hidden)
                .view(batch, source_width, self.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
            key = torch.cat((source_key, slot_key), dim=-2)
            value = torch.cat((source_value, slot_value), dim=-2)

        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=keep_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
            scale=self.head_dim**-0.5,
        )
        attended = attended.transpose(1, 2).reshape(batch, slots, -1)
        return self.o_proj(attended)



class GrepSeekDFlashDecoderLayer(nn.Module):
    """A parameter-shape-isomorphic Qwen3 DFlash decoder layer."""

    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = GrepSeekDFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        source_hidden: Optional[torch.Tensor],
        keep_mask: torch.Tensor,
        slot_cos: torch.Tensor,
        slot_sin: torch.Tensor,
        source_cos: Optional[torch.Tensor],
        source_sin: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            normalized,
            source_hidden,
            keep_mask,
            slot_cos,
            slot_sin,
            source_cos,
            source_sin,
        )
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))



class CompressedContextOutput(NamedTuple):
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    slot_counts: torch.Tensor



class GrepSeekDFlashCompressor(nn.Module):
    """Variable-length 1:N compressor with native DFlash backbone shapes."""

    def __init__(
        self,
        draft_config: Qwen3Config,
        *,
        target_hidden_size: int,
        target_layer_ids: Sequence[int] = DEFAULT_QWEN35_TARGET_LAYER_IDS,
        compression_ratio: int = 16,
        local_window: int = 256,
        output_embedding_rms: float = 0.02,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        num_layers = int(draft_config.num_hidden_layers)
        if num_layers < 2:
            raise ValueError("the DFlash compressor must have at least two layers")
        if int(draft_config.hidden_size) != int(target_hidden_size):
            raise ValueError(
                "draft and target hidden sizes must match for continuous-token recovery"
            )
        ids = tuple(int(layer_id) for layer_id in target_layer_ids)
        if not ids or len(set(ids)) != len(ids) or any(i < 0 for i in ids):
            raise ValueError("target_layer_ids must be unique nonnegative integers")
        for name, value in (
            ("target_hidden_size", target_hidden_size),
            ("compression_ratio", compression_ratio),
            ("local_window", local_window),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if local_window < compression_ratio:
            raise ValueError(
                "local_window must be at least compression_ratio so every source "
                "token is visible to its compression slot"
            )
        if not math.isfinite(float(output_embedding_rms)) or output_embedding_rms <= 0:
            raise ValueError("output_embedding_rms must be finite and positive")

        self.config = copy.deepcopy(draft_config)
        self.target_hidden_size = int(target_hidden_size)
        self.target_layer_ids = ids
        self.compression_ratio = int(compression_ratio)
        self.local_window = int(local_window)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        hidden_size = int(self.config.hidden_size)

        # These names and matrix shapes match native DFlash exactly.
        self.fc = nn.Linear(len(ids) * self.target_hidden_size, hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(
            hidden_size, eps=float(self.config.rms_norm_eps)
        )
        self.layers = nn.ModuleList(
            [
                GrepSeekDFlashDecoderLayer(self.config, layer_idx)
                for layer_idx in range(num_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(hidden_size, eps=float(self.config.rms_norm_eps))
        self.rotary_emb = Qwen3RotaryEmbedding(self.config)

        # Native DFlash receives repeated target MASK embeddings.  Here the
        # same single seed is trainable because only the compressor is trained;
        # RoPE at evidence anchors breaks the otherwise exact symmetry.
        self.mask_embedding = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.register_buffer(
            "output_embedding_rms",
            torch.tensor(float(output_embedding_rms), dtype=torch.float32),
            persistent=True,
        )

        initializer_range = float(getattr(self.config, "initializer_range", 0.02))
        self.apply(lambda module: self._initialize_module(module, initializer_range))
        nn.init.normal_(self.mask_embedding, mean=0.0, std=initializer_range)

    @staticmethod
    def _initialize_module(module: nn.Module, std: float) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)




    def forward(
        self,
        source_features: torch.Tensor,
        source_attention_mask: torch.Tensor,
        source_token_lengths: Optional[torch.Tensor] = None,
        slot_token_lengths: Optional[torch.Tensor] = None,
    ) -> CompressedContextOutput:
        if source_features.ndim != 3 or not torch.is_floating_point(source_features):
            raise ValueError(
                "source_features must be floating [segments, source, selected_hidden]"
            )
        if source_attention_mask.shape != source_features.shape[:2]:
            raise ValueError("source feature and attention-mask shapes differ")
        expected_width = len(self.target_layer_ids) * self.target_hidden_size
        if source_features.shape[-1] != expected_width:
            raise ValueError(
                f"source feature width must be {expected_width}, got "
                f"{source_features.shape[-1]}"
            )
        if source_features.device != self.fc.weight.device:
            raise ValueError("source features and compressor must share a device")

        source_features = source_features.to(dtype=self.fc.weight.dtype)
        layout = build_local_compression_layout(
            source_attention_mask,
            compression_ratio=self.compression_ratio,
            local_window=self.local_window,
            source_token_lengths=source_token_lengths,
            slot_token_lengths=slot_token_lengths,
        )
        batch, slot_width = layout.slot_attention_mask.shape
        if slot_width == 0:
            empty = source_features.new_zeros((batch, 0, int(self.config.hidden_size)))
            return CompressedContextOutput(
                empty, layout.slot_attention_mask, layout.slot_counts
            )

        source_hidden = self.hidden_norm(self.fc(source_features))
        hidden_states = self.mask_embedding.expand(batch, slot_width, -1)
        slot_cos, slot_sin = self.rotary_emb(hidden_states, layout.slot_position_ids)
        source_cos, source_sin = self.rotary_emb(
            source_hidden, layout.source_position_ids
        )
        slot_row_mask = layout.slot_attention_mask.unsqueeze(-1)

        # Local overlapping reads, exactly followed by one global slot-only
        # mixer. Invalid rows are zero after every residual block.  This maps
        # public 6-layer Qwen3.5/Qwen3.6-35B-A3B drafts to 5+1 and the public
        # 5-layer Qwen3.6-27B draft to 4+1 without changing parameter geometry.
        for layer in self.layers[:-1]:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    source_hidden,
                    layout.local_keep_mask,
                    slot_cos,
                    slot_sin,
                    source_cos,
                    source_sin,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    source_hidden,
                    layout.local_keep_mask,
                    slot_cos,
                    slot_sin,
                    source_cos,
                    source_sin,
                )
            hidden_states = hidden_states * slot_row_mask

        final_layer = self.layers[-1]
        if self.gradient_checkpointing and self.training:
            hidden_states = checkpoint(
                final_layer,
                hidden_states,
                None,
                layout.global_keep_mask,
                slot_cos,
                slot_sin,
                None,
                None,
                use_reentrant=False,
            )
        else:
            hidden_states = final_layer(
                hidden_states,
                None,
                layout.global_keep_mask,
                slot_cos,
                slot_sin,
                None,
                None,
            )
        hidden_states = hidden_states * slot_row_mask
        scale = self.output_embedding_rms.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        embeddings = self.norm(hidden_states) * scale
        embeddings = embeddings * slot_row_mask
        return CompressedContextOutput(
            embeddings=embeddings,
            attention_mask=layout.slot_attention_mask,
            slot_counts=layout.slot_counts,
        )

    def export_config(self) -> dict[str, Any]:
        return {
            "architecture": type(self).__name__,
            "target_layer_ids": list(self.target_layer_ids),
            "target_hidden_size": self.target_hidden_size,
            "selected_source_width": len(self.target_layer_ids)
            * self.target_hidden_size,
            "compression_ratio": self.compression_ratio,
            "local_window": self.local_window,
            "hidden_size": int(self.config.hidden_size),
            "intermediate_size": int(self.config.intermediate_size),
            "num_hidden_layers": int(self.config.num_hidden_layers),
            "num_attention_heads": int(self.config.num_attention_heads),
            "num_key_value_heads": int(self.config.num_key_value_heads),
            "head_dim": int(self.config.head_dim),
            "layer_types": ["sliding_attention"]
            * (int(self.config.num_hidden_layers) - 1)
            + ["full_attention"],
            "gradient_checkpointing": self.gradient_checkpointing,
            "output_embedding_rms": float(self.output_embedding_rms.item()),
            "slot_padding_side": "left",
        }
