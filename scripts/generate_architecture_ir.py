#!/usr/bin/env python3
"""Generate a layout-free Architecture IR from an official saved config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from architecture_ir import extract, source_ir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--config-url", required=True)
    parser.add_argument("--implementation-url", required=True)
    parser.add_argument("--architecture", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sources = source_ir(args.repository, args.config_url, args.implementation_url, args.architecture)
    args.output.write_text(json.dumps(extract(config, sources), indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
