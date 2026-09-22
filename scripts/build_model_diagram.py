#!/usr/bin/env python3
"""Run the reproducible model-diagram build, check and preview workflow."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from validate_diagram_spec import validate


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def compact_unit(value: int) -> str:
    """Render a model dimension with binary K/M units, matching diagram labels."""
    divisor, suffix = (1024 * 1024, "M") if value >= 1024 * 1024 else (1024, "K")
    rendered = f"{value / divisor:.2f}".rstrip("0").rstrip(".")
    return f"{rendered}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path, help="diagram specification JSON")
    parser.add_argument("--skip-preview", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec_path = args.spec.resolve()
    spec = validate(spec_path)

    renderer = root / spec["renderer"]
    if not renderer.is_file():
        raise FileNotFoundError(f"renderer not found: {renderer}")
    run([sys.executable, str(renderer)], root)

    output_dir = root / spec["output_directory"]
    svg = output_dir / "architecture.svg"
    if not svg.is_file():
        raise FileNotFoundError(f"renderer did not create {svg}")
    run([sys.executable, "scripts/check_svg_arrow_connections.py", str(svg)], root)

    svg_text = svg.read_text(encoding="utf-8")
    presentations = spec.get("parameter_presentation", {})
    for key, value in spec["required_parameters"].items():
        presentation = presentations.get(key)
        if presentation is None:
            expected = f"{key} = {value:,}"
        else:
            rendered = compact_unit(value)
            if presentation.get("format") == "compact_tokens":
                rendered += " tokens"
            elif presentation.get("format") != "compact":
                raise ValueError(f"unsupported parameter format for {key}: {presentation.get('format')}")
            expected = f"{presentation['label']} · {rendered}"
        if expected not in svg_text:
            raise ValueError(f"diagram is missing required rendered parameter: {expected}")

    if not args.skip_preview:
        chrome = shutil.which("google-chrome") or shutil.which("chromium")
        mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        if chrome is None and mac_chrome.is_file():
            chrome = str(mac_chrome)
        if chrome is None:
            raise RuntimeError("Chrome/Chromium is required for PNG preview; use --skip-preview only in CI")
        canvas = spec.get("canvas", {})
        width, height = canvas.get("width", 2200), canvas.get("height", 1650)
        png = output_dir / "architecture.png"
        run([
            chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
            f"--window-size={width},{height}", f"--screenshot={png}", svg.resolve().as_uri(),
        ], root)
        if not png.is_file() or png.stat().st_size < 10_000:
            raise RuntimeError(f"preview render failed: {png}")

    manifest = {
        "model_name": spec["model_name"],
        "repository": spec["repository"],
        "spec": str(spec_path.relative_to(root)),
        "config": spec["config_file"],
        "svg": str(svg.relative_to(root)),
        "png": str((output_dir / "architecture.png").relative_to(root)),
        "sources": spec.get("sources", {}),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: diagram bundle written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
