"""The six-layer, 1.24B GLM memory encoder used in local evaluations."""
from transformers import Qwen3Config

from .layers import GrepSeekDFlashCompressor


def build_compressor(config):
    c = config["compressor"]
    draft = Qwen3Config(
        vocab_size=154880, hidden_size=4096, intermediate_size=12288,
        num_hidden_layers=6, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, hidden_act="silu", max_position_embeddings=262144,
        initializer_range=0.02, rms_norm_eps=1e-6, use_cache=True,
        tie_word_embeddings=False, attention_bias=False, attention_dropout=0.0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0},
        use_sliding_window=True, sliding_window=4096, max_window_layers=6,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
    )
    return GrepSeekDFlashCompressor(draft, target_hidden_size=4096,
        target_layer_ids=c["target_layer_ids"], compression_ratio=16,
        local_window=c["local_window"], output_embedding_rms=c["output_embedding_rms"],
        gradient_checkpointing=False)
