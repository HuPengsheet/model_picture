#!/usr/bin/env python3
"""Compose slot-based SVG templates from Architecture IR and DiagramSpec.

DiagramSpec may refer to a structural IR value with:
    {"ref": "parameters.hidden_size", "format": "compact"}
This prevents presentation configuration from duplicating model facts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from svg_template import instantiate, viewbox_size


def resolve_ref(document: dict, ref: str):
    value = document
    for part in ref.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"IR reference does not exist: {ref}")
        value = value[part]
    return value


def compact(value: int) -> str:
    divisor, suffix = (1024 * 1024, "M") if value >= 1024 * 1024 else (1024, "K")
    return f"{value / divisor:.2f}".rstrip("0").rstrip(".") + suffix


def resolve_slots(ir: dict, declared_slots: dict) -> dict[str, str]:
    slots = {}
    for name, value in declared_slots.items():
        if not isinstance(value, dict) or "ref" not in value:
            slots[name] = str(value)
            continue
        resolved = resolve_ref(ir, value["ref"])
        if value.get("format") == "compact":
            resolved = compact(resolved)
        elif value.get("format") not in (None, "raw"):
            raise ValueError(f"unsupported slot format: {value['format']}")
        slots[name] = str(resolved)
    return slots


def transform_for_instance(template: Path, instance: dict) -> dict:
    """Resolve explicit x/y or anchor-target placement into an SVG transform."""
    transform = instance.get("transform", {})
    placement = instance.get("placement")
    if placement is None:
        required = {"x", "y", "width", "height"}
        if not required <= transform.keys():
            raise ValueError(f"instance {instance['id']} needs transform {sorted(required)}")
        return transform
    required = {"anchor", "target", "width", "height"}
    if not required <= placement.keys():
        raise ValueError(f"instance {instance['id']} needs placement {sorted(required)}")
    _fragment, anchors = instantiate(template, "anchor_probe", {}, 0, 0, placement["width"], placement["height"])
    anchor_name = placement["anchor"]
    if anchor_name not in anchors:
        raise ValueError(f"template {template.name} has no anchor {anchor_name!r}")
    target = placement["target"]
    if not {"x", "y"} <= target.keys():
        raise ValueError(f"instance {instance['id']} placement.target needs x and y")
    view_width, view_height = viewbox_size(template)
    anchor_x, anchor_y = anchors[anchor_name]
    return {
        "x": target["x"] - anchor_x * placement["width"] / view_width,
        "y": target["y"] - anchor_y * placement["height"] / view_height,
        "width": placement["width"],
        "height": placement["height"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ir", type=Path)
    parser.add_argument("diagram_spec", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ir = json.loads(args.ir.read_text(encoding="utf-8"))
    spec = json.loads(args.diagram_spec.read_text(encoding="utf-8"))
    canvas = spec["canvas"]
    fragments = []
    for instance in spec["instances"]:
        template = root / instance["template"]
        transform = transform_for_instance(template, instance)
        fragment, _anchors = instantiate(
            template, instance["id"], resolve_slots(ir, instance.get("slots", {})),
            transform["x"], transform["y"], transform["width"], transform["height"],
        )
        fragments.append(fragment)
    title = escape(spec.get("title", ir["model"]["name"]))
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas["width"]}" height="{canvas["height"]}" '
        f'viewBox="0 0 {canvas["width"]} {canvas["height"]}"><title>{title}</title>'
        f'<rect width="{canvas["width"]}" height="{canvas["height"]}" fill="white"/>'
        + "".join(fragments) + "</svg>"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
