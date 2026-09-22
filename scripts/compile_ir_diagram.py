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

from svg_template import instantiate


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
        transform = instance["transform"]
        fragment, _anchors = instantiate(
            root / instance["template"], instance["id"], resolve_slots(ir, instance.get("slots", {})),
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
