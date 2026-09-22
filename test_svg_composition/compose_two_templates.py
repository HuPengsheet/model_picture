#!/usr/bin/env python3
"""Anchor-based composition smoke test for two independent SVG templates.

Run from the repository root:
    python3 test_svg_composition/compose_two_templates.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from svg_template import instantiate, viewbox_size  # noqa: E402


def place_on_anchor(template: Path, anchor: str, target: tuple[float, float], width: float, height: float) -> dict[str, float]:
    """Compute an embedded SVG transform so one local anchor reaches target."""
    _probe, anchors = instantiate(template, "probe", {}, 0, 0, width, height)
    local_x, local_y = anchors[anchor]
    view_width, view_height = viewbox_size(template)
    return {
        "x": target[0] - local_x * width / view_width,
        "y": target[1] - local_y * height / view_height,
        "width": width,
        "height": height,
    }


def main() -> int:
    output = ROOT / "test_svg_composition/combined.svg"
    swiglu = ROOT / "artifacts/diagrams/templates/components/SwiGLU.svg"
    gated_attention = ROOT / "artifacts/diagrams/templates/components/gate_attention.svg"

    # Both components use their `input` anchors as the placement reference.
    # Moving either component only requires changing its target x/y below.
    swiglu_transform = place_on_anchor(swiglu, "input", (360, 650), 550, 270)
    attention_transform = place_on_anchor(gated_attention, "input", (350+550, 650), 450, 325)

    swiglu_svg, _ = instantiate(swiglu, "swiglu", {}, **swiglu_transform)
    attention_svg, _ = instantiate(gated_attention, "gated_attention", {}, **attention_transform)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="800" viewBox="0 0 1500 800">
  <title>Two-template SVG composition test</title>
  <desc>SwiGLU and Gated Attention independently placed by their input anchors.</desc>
  <rect width="1500" height="800" fill="#eaf1f8"/>
  <text x="750" y="55" text-anchor="middle" font-family="Arial,sans-serif" font-size="30" font-weight="700">Anchor-based SVG composition</text>
  <path d="M80 650 H1420" fill="none" stroke="#8a8a8a" stroke-width="1.5" stroke-dasharray="7 7"/>
  <text x="95" y="635" font-family="Arial,sans-serif" font-size="16" fill="#666">shared input-anchor baseline</text>
  {swiglu_svg}
  {attention_svg}
</svg>'''
    output.write_text(svg, encoding="utf-8")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_svg_arrow_connections.py"), str(output)],
        check=True,
        cwd=ROOT,
    )
    print(f"PASS {output}")
    print(f"SwiGLU input target: (360, 650), outer origin: ({swiglu_transform['x']:.1f}, {swiglu_transform['y']:.1f})")
    print(f"Gated Attention input target: (1080, 650), outer origin: ({attention_transform['x']:.1f}, {attention_transform['y']:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
