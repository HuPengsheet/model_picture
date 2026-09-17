#!/usr/bin/env python3
"""Normalize downloaded Qwen Hugging Face configs into diagram-ready structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def present(config: dict, names: tuple[str, ...]) -> dict:
    return {name: config[name] for name in names if name in config}


def compact_layer_types(layer_types: list[str]) -> list[dict]:
    """Turn [linear, linear, full, ...] into diagram-friendly contiguous spans."""
    spans: list[dict] = []
    for index, kind in enumerate(layer_types):
        if spans and spans[-1]["type"] == kind and spans[-1]["last_layer"] == index - 1:
            spans[-1]["last_layer"] = index
        else:
            spans.append({"first_layer": index, "last_layer": index, "type": kind})
    return spans


def normalize(raw: dict, source: Path) -> dict:
    # Native multimodal Qwen3.5 stores the language-transformer topology under
    # text_config. Keep the wrapper architecture separately and describe the
    # actual text backbone from its nested config.
    text = raw.get("text_config", raw)
    result = {
        "source_config": str(source),
        "evidence": "Fields are copied from Hugging Face config.json; absent fields are not inferred.",
        "architecture": {
            "wrapper_architectures": raw.get("architectures"),
            "wrapper_model_type": raw.get("model_type"),
            "backbone_architectures": text.get("architectures"),
            "backbone_model_type": text.get("model_type"),
            "text_config_is_nested": "text_config" in raw,
        },
        "backbone": present(text, ("vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size", "hidden_act")),
        "attention": present(text, (
            "num_attention_heads", "num_key_value_heads", "head_dim", "attention_bias",
            "attention_dropout", "attn_dropout_prob", "sliding_window", "use_sliding_window",
            "max_window_layers", "full_attention_interval", "linear_conv_kernel_dim",
            "linear_key_head_dim", "linear_num_key_heads", "linear_num_value_heads",
            "linear_value_head_dim", "attn_output_gate",
        )),
        "mlp_or_moe": present(text, (
            "intermediate_size", "hidden_act", "num_experts", "num_experts_per_tok",
            "moe_intermediate_size", "shared_expert_intermediate_size", "num_shared_experts",
            "router_aux_loss_coef", "norm_topk_prob", "decoder_sparse_step", "mlp_only_layers",
        )),
        "normalization_and_residual": present(text, ("rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps")),
        "position_encoding": present(text, (
            "max_position_embeddings", "rope_theta", "rope_scaling", "rope_parameters",
            "rotary_emb_base", "rotary_pct", "use_dynamic_ntk", "use_logn_attn",
        )),
        "input_output": present(text, ("vocab_size", "tie_word_embeddings", "bos_token_id", "eos_token_id")),
        "runtime_cache": present(text, ("use_cache",)),
    }
    layer_types = text.get("layer_types")
    result["layer_schedule"] = (
        {"source": "config.layer_types", "layers": compact_layer_types(layer_types)}
        if isinstance(layer_types, list)
        else {"source": "not declared in config", "layers": None}
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/hf_qwen_configs"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen_model_structures"))
    args = parser.parse_args()
    configs = sorted(path for path in args.input.rglob("*.json") if path.name != "manifest.json")
    index = []
    for config_path in configs:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        structure = normalize(raw, config_path)
        relative = config_path.relative_to(args.input)
        out_path = args.output / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(structure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index.append({
            "model": f"Qwen/{config_path.stem}",
            "structure_file": str(out_path),
            "architecture": structure["architecture"],
            "layer_schedule_source": structure["layer_schedule"]["source"],
        })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(index)} diagram-ready structure files to {args.output}")


if __name__ == "__main__":
    main()
