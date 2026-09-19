#!/usr/bin/env python3
"""Validate a model diagram specification against its saved official config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_PARAMETERS = ("hidden_size", "vocab_size", "max_position_embeddings")
SUPPORTED_COMPONENTS = {
    "multi_head_attention", "sliding_window_attention", "gqa", "gated_attention",
    "gated_delta_rule", "gated_gqa", "swiglu", "sparse_moe",
}


def fail(message: str) -> None:
    raise ValueError(message)


def validate(spec_path: Path) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    root = spec_path.resolve().parents[2]
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
    for spec_key, config_key in (
        ("num_experts", "num_experts"),
        ("num_experts_per_token", "num_experts_per_tok"),
    ):
        if spec_key in backbone and backbone[spec_key] != text_config.get(config_key):
            fail(f"{spec_key} does not match config.{config_key}")
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
