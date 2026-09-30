#!/usr/bin/env python3
"""Generate the window_size=100 example entirely through IR + template config."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "test_svg_composition"
OUTPUT = TEST / "sliding_window_100.svg"

subprocess.run([
    sys.executable, str(ROOT / "scripts/compile_ir_diagram.py"),
    str(TEST / "sliding_window_100_ir.json"),
    str(TEST / "sliding_window_100_diagram_spec.json"),
    str(OUTPUT),
], check=True, cwd=ROOT)
subprocess.run([sys.executable, str(ROOT / "scripts/check_svg_arrow_connections.py"), str(OUTPUT)], check=True, cwd=ROOT)
print(f"PASS {OUTPUT}")
