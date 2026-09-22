"""Canonical, layout-free architecture IR extractors.

The IR deliberately contains only model facts and execution structure.  SVG
templates, coordinates, colours and labels belong to DiagramSpec, never here.
"""
from __future__ import annotations

from typing import Any


IR_VERSION = "1.0"


def source_ir(repository: str, config_url: str, implementation_url: str, architecture: str) -> dict[str, Any]:
    return {
        "repository": repository,
        "config_url": config_url,
        "implementation_url": implementation_url,
        "architecture_class": architecture,
    }


def mimo_v2_ir(config: dict[str, Any], sources: dict[str, Any]) -> dict[str, Any]:
    """Extract MiMo V2 facts from its official configuration schema."""
    hybrid = config["hybrid_layer_pattern"]
    moe_frequency = config["moe_layer_freq"]
    full_layers = [index + 1 for index, value in enumerate(hybrid) if value == 0]
    sliding_layers = [index + 1 for index, value in enumerate(hybrid) if value == 1]
    dense_layers = [index + 1 for index, value in enumerate(moe_frequency) if value == 0]
    moe_layers = [index + 1 for index, value in enumerate(moe_frequency) if value == 1]
    return {
        "ir_version": IR_VERSION,
        "model": {
            "name": "MiMo-V2.6-Pro-RL",
            "architecture": config["architectures"][0],
            "model_type": "multimodal_decoder",
        },
        "parameters": {
            "hidden_size": config["hidden_size"],
            "vocab_size": config["vocab_size"],
            "context_length": config["max_position_embeddings"],
        },
        "io": {
            "input_type": ["text", "image", "audio"],
            "embedding": {"type": "token_embedding", "dimension": config["hidden_size"]},
            "lm_head": {"type": "linear", "tied": config["tie_word_embeddings"]},
        },
        "backbone": {
            "kind": "decoder_only",
            "num_layers": config["num_hidden_layers"],
            "final_norm": {"type": "rms_norm", "epsilon": config["layernorm_epsilon"]},
            "default_block": {
                "execution_order": [
                    "input_rms_norm", "attention", "residual_add",
                    "post_attention_rms_norm", "ffn", "residual_add",
                ],
                "residuals": 2,
            },
            "attention_variants": {
                "full_attention": {
                    "type": "gqa", "causal": True,
                    "num_attention_heads": config["num_attention_heads"],
                    "num_key_value_heads": config["num_key_value_heads"],
                    "qk_head_dim": config["head_dim"], "v_head_dim": config["v_head_dim"],
                    "position": {"type": "rope", "theta": config["rope_theta"], "partial_factor": config["partial_rotary_factor"]},
                },
                "sliding_window_attention": {
                    "type": "gqa", "causal": True, "window_size": config["sliding_window"],
                    "num_attention_heads": config["swa_num_attention_heads"],
                    "num_key_value_heads": config["swa_num_key_value_heads"],
                    "qk_head_dim": config["swa_head_dim"], "v_head_dim": config["swa_v_head_dim"],
                    "position": {"type": "rope", "theta": config["swa_rope_theta"], "partial_factor": config["partial_rotary_factor"]},
                },
            },
            "ffn_variants": {
                "dense_swiglu": {
                    "type": "swiglu", "activation": config["hidden_act"],
                    "intermediate_size": config["intermediate_size"],
                },
                "sparse_moe": {
                    "type": "sparse_moe", "activation": config["hidden_act"],
                    "routed_experts": config["n_routed_experts"],
                    "shared_experts": config["n_shared_experts"] or 0,
                    "top_k": config["num_experts_per_tok"],
                    "router": {"score_function": config["scoring_func"], "topk_method": config["topk_method"], "groups": config["n_group"]},
                    "expert_intermediate_size": config["moe_intermediate_size"],
                },
            },
            "layer_schedule": [
                {"slot": "attention", "variant": "full_attention", "layer_ids": full_layers},
                {"slot": "attention", "variant": "sliding_window_attention", "layer_ids": sliding_layers},
                {"slot": "ffn", "variant": "dense_swiglu", "layer_ids": dense_layers},
                {"slot": "ffn", "variant": "sparse_moe", "layer_ids": moe_layers},
            ],
        },
        "multimodal": {
            "vision_encoder": {
                "kind": "vision_transformer", "num_layers": config["vision_config"]["depth"],
                "hidden_size": config["vision_config"]["hidden_size"], "output_size": config["vision_config"]["out_hidden_size"],
            },
            "audio_encoder": {
                "kind": "audio_encoder", "num_layers": config["audio_config"]["input_local_layers"],
                "output_size": config["audio_config"]["out_hidden_size"],
            },
            "fusion": {"type": "token_embedding_replacement"},
        },
        "sources": sources,
    }


EXTRACTORS = {"mimo_v2": mimo_v2_ir}


def extract(config: dict[str, Any], sources: dict[str, Any]) -> dict[str, Any]:
    try:
        return EXTRACTORS[config["model_type"]](config, sources)
    except KeyError as error:
        raise ValueError(f"no Architecture IR extractor for model_type={config.get('model_type')!r}") from error
