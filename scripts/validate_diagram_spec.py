#!/usr/bin/env python3
"""Validate a model diagram specification against its saved official config."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED_PARAMETERS = ("hidden_size", "vocab_size", "max_position_embeddings")
SUPPORTED_COMPONENTS = {
    "multi_head_attention", "sliding_window_attention", "gqa", "gated_attention",
    "gated_delta_rule", "gated_gqa", "mla", "dense_ffn", "swiglu", "sparse_moe",
}


def fail(message: str) -> None:
    raise ValueError(message)


def validate(spec_path: Path) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    for key in ("model_name", "repository", "config_file", "output_directory", "renderer"):
        if not spec.get(key):
            fail(f"missing required spec field: {key}")

    config_path = root / spec["config_file"]
    if not config_path.is_file():
        fail(f"saved config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)

    declared = spec.get("required_parameters", {})
    for key in REQUIRED_PARAMETERS:
        if key not in text_config:
            fail(f"official config does not declare required parameter: {key}")
        if declared.get(key) != text_config[key]:
            fail(f"{key}: spec={declared.get(key)!r}, config={text_config[key]!r}")

    backbone = spec.get("text_backbone", {})
    if backbone.get("num_hidden_layers") != text_config.get("num_hidden_layers"):
        fail("num_hidden_layers does not match config")
    pattern = backbone.get("layer_pattern", [])
    repetitions = backbone.get("pattern_repetitions")
    expected_layers = pattern * repetitions if pattern and isinstance(repetitions, int) else None
    if expected_layers and expected_layers != text_config.get("layer_types"):
        fail("expanded layer_pattern does not match config.layer_types")
    if expected_layers and len(expected_layers) != backbone["num_hidden_layers"]:
        fail("expanded layer pattern length does not match num_hidden_layers")

    unknown = set(backbone.get("components", [])) - SUPPORTED_COMPONENTS
    if unknown:
        fail(f"unsupported component names: {', '.join(sorted(unknown))}")
    for spec_key, config_keys in (
        ("num_experts", ("num_experts", "n_routed_experts")),
        ("num_experts_per_token", ("num_experts_per_tok",)),
    ):
        configured = next((text_config[key] for key in config_keys if key in text_config), None)
        if spec_key in backbone and backbone[spec_key] != configured:
            fail(f"{spec_key} does not match config ({'/'.join(config_keys)})")

    # A drawing config may select reusable templates. Validate these paths here
    # so a renderer never silently falls back to hand-drawn, untracked assets.
    template_plan = spec.get("template_plan")
    if template_plan is not None:
        main_template = template_plan.get("main_network")
        components = template_plan.get("components", [])
        if not isinstance(main_template, str) or not main_template:
            fail("template_plan.main_network must be a non-empty template path")
        if not isinstance(components, list) or not all(isinstance(path, str) and path for path in components):
            fail("template_plan.components must be a list of non-empty template paths")
        for template_path in (main_template, *components):
            if not (root / template_path).is_file():
                fail(f"selected template does not exist: {template_path}")

    # Labels are part of the drawing config, not free-form copies of model
    # dimensions. Check the two labels that can be derived generically.
    labels = spec.get("labels", {})
    if "embedding_dimension" in labels and labels["embedding_dimension"] != text_config["hidden_size"]:
        fail("labels.embedding_dimension does not match hidden_size")
    if "repeat_label" in labels:
        match = re.search(r"\d+", str(labels["repeat_label"]))
        if match is None or int(match.group()) != text_config["num_hidden_layers"]:
            fail("labels.repeat_label does not match num_hidden_layers")
    if "dense_layers" in labels:
        expected_dense = text_config.get("first_k_dense_replace")
        if expected_dense is None and "moe_layer_freq" in text_config:
            expected_dense = text_config["moe_layer_freq"].count(0)
        if expected_dense is None or labels["dense_layers"] != expected_dense:
            fail("labels.dense_layers does not match the Dense/MoE schedule")
    if "moe_layers" in labels:
        expected_dense = text_config.get("first_k_dense_replace")
        if expected_dense is None and "moe_layer_freq" in text_config:
            expected_dense = text_config["moe_layer_freq"].count(0)
        if expected_dense is None or labels["moe_layers"] != text_config["num_hidden_layers"] - expected_dense:
            fail("labels.moe_layers does not match the Dense/MoE schedule")
    for label_key, pattern_value in (("full_attention_layers", 0), ("sliding_window_layers", 1)):
        if label_key in labels:
            pattern = text_config.get("hybrid_layer_pattern")
            if not isinstance(pattern, list) or labels[label_key] != pattern.count(pattern_value):
                fail(f"labels.{label_key} does not match hybrid_layer_pattern")
    return spec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    spec = validate(args.spec)
    print(f"PASS {args.spec}: {spec['model_name']} specification matches saved config")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
