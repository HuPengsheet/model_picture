#!/usr/bin/env python3
"""Fill a main-network SVG template from a semantic template configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from svg_template import instantiate, viewbox_size


def attention_schedule_label(attention: dict) -> str:
    if attention.get("schedule_label"):
        return attention["schedule_label"]
    if attention.get("type") != "hybrid":
        return attention.get("schedule_label", "causal attention")
    variants = attention.get("variants", [])
    names = []
    for variant in variants:
        name = variant["type"].replace("_", " ")
        if "window_size" in variant:
            name += f" · w={variant['window_size']}"
        names.append(name)
    return " + ".join(names) if names else "hybrid attention"


def slots_from_config(config: dict) -> dict[str, str]:
    model, decoder = config["model"], config["decoder"]
    attention, ffn = decoder["attention"], decoder["ffn"]
    return {
        "lm_head_label": model["lm_head"].get("label", "LM Head"),
        "final_norm_label": f"Final {model['final_norm']['type']}",
        "decoder_repeat_label": model.get("repeat_label", f"Decoder Block × {model['num_layers']}"),
        "input_norm_label": decoder["input_norm"]["type"],
        "post_attention_norm_label": decoder["post_attention_norm"]["type"],
        "attention_label": attention.get("label", attention["type"].replace("_", " ").title()),
        "attention_schedule_label": attention_schedule_label(attention),
        "ffn_label": ffn.get("label", ffn["type"].replace("_", " ").upper()),
        "embedding_label": model["embedding_label"],
        "input_label": model["input_label"],
        "residual_operator": decoder.get("residual_operator", "+"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    template = root / config["template"]
    width, height = viewbox_size(template)
    fragment, _anchors = instantiate(
        template, "main_network", slots_from_config(config), 0, 0, width, height,
        set(config.get("hidden_roles", [])),
    )
    title = config.get("title", "Main network architecture")
    description = config.get("description", "")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" height="{height:g}" '
        f'viewBox="0 0 {width:g} {height:g}"><title>{title}</title><desc>{description}</desc>'
        f'{fragment}</svg>'
    )
    args.output.write_text(svg, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
